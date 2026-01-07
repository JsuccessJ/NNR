import os
import signal
import shutil
import json
from config import Config
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_Train_Dataset
from util import AvgMetric
from util import compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
        self.model = model # model.py에 forward() 메서드가 있음
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid

        # PLM 기반 모델의 경우 learning rate 분리
        if config.use_plm_news_encoder:
            # PLM 파라미터와 나머지 파라미터 분리
            plm_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'plm' in name:  # PLM 파라미터
                        plm_params.append(param)
                    else:  # 다른 파라미터 (category embedding, attention 등)
                        other_params.append(param)

            # 서로 다른 learning rate 설정
            self.optimizer = optim.Adam([
                {'params': plm_params, 'lr': config.plm_lr},      # 1e-5 (PLM)
                {'params': other_params, 'lr': config.lr}         # 1e-4 (기타)
            ], weight_decay=config.weight_decay)

            print(f'[Single-GPU] PLM parameters: {len(plm_params)}, Other parameters: {len(other_params)}')
            print(f'[Single-GPU] PLM lr: {config.plm_lr}, Other lr: {config.lr}')
        else:
            # 기존 방식 (non-PLM 모델)
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.lr,
                weight_decay=config.weight_decay
            )

        self._dataset = config.dataset
        self.mind_corpus = mind_corpus
        self.train_dataset = MIND_Train_Dataset(mind_corpus)

        # 전체 training step 계산
        total_steps = len(self.train_dataset) // config.batch_size * config.epoch
        warmup_steps = int(total_steps * 0.1)  # 10% warmup

        # LR Scheduler 추가 (논문: 10% warmup + linear decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.run_index = run_index
        self.model_dir = config.model_dir + '/#' + str(self.run_index)
        self.best_model_dir = config.best_model_dir + '/#' + str(self.run_index)
        self.dev_res_dir = config.dev_res_dir + '/#' + str(self.run_index)
        self.result_dir = config.result_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.best_model_dir):
            os.mkdir(self.best_model_dir)
        if not os.path.exists(self.dev_res_dir):
            os.mkdir(self.dev_res_dir)
        with open(config.config_dir + '/#' + str(self.run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if self._dataset == 'large':
            self.prediction_dir = config.prediction_dir + '/#' + str(self.run_index)
            os.mkdir(self.prediction_dir)
        self.dev_criterion = config.dev_criterion
        self.early_stopping_epoch = config.early_stopping_epoch
        self.auc_results = []
        self.mrr_results = []
        self.ndcg5_results = []
        self.ndcg10_results = []
        self.best_dev_epoch = 0
        self.best_dev_auc = 0
        self.best_dev_mrr = 0
        self.best_dev_ndcg5 = 0
        self.best_dev_ndcg10 = 0
        self.best_dev_avg = AvgMetric(0, 0, 0, 0)
        self.epoch_not_increase = 0
        self.gradient_clip_norm = config.gradient_clip_norm
        self.model.cuda()
        print('Running : ' + self.model.model_name + '\t#' + str(self.run_index))

    def negative_log_softmax(self, logits):
        loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
        return loss

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss

    def train(self):
        model = self.model
        for e in tqdm(range(1, self.epoch + 1), desc='Epoch'):
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            model.train()
            epoch_loss = 0
            
            # 배치 단위 진행률 표시
            train_dataloader_with_progress = tqdm(train_dataloader, desc=f'Epoch {e}/{self.epoch}', leave=False)
            
            for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) in train_dataloader_with_progress:
                user_ID = user_ID.cuda(non_blocking=True)                                                                                                                       # [batch_size]
                user_category = user_category.cuda(non_blocking=True)                                                                                                           # [batch_size, max_history_num]
                user_subCategory = user_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, max_history_num]
                user_title_text = user_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_mask = user_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
                user_title_entity = user_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_title_length]
                user_content_text = user_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_mask = user_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
                user_content_entity = user_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, max_history_num, max_content_length]
                user_history_mask = user_history_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num]
                user_history_graph = user_history_graph.cuda(non_blocking=True)                                                                                                 # [batch_size, max_history_num, max_history_num]
                user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)                                                                                 # [batch_size, category_num + 1]
                user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)                                                                           # [batch_size, max_history_num]
                news_category = news_category.cuda(non_blocking=True)                                                                                                           # [batch_size, 1 + negative_sample_num]
                news_subCategory = news_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, 1 + negative_sample_num]
                news_title_text = news_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_mask = news_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_entity = news_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_title_length]
                news_content_text = news_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_mask = news_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
                news_content_entity = news_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, 1 + negative_sample_num, max_content_length]

                logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                               news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) # [batch_size, 1 + negative_sample_num], # model.py:120-133에 forward() 메서드가 있음

                loss = self.loss(logits)
                if model.news_encoder.auxiliary_loss is not None:
                    news_auxiliary_loss = model.news_encoder.auxiliary_loss.mean()
                    loss += news_auxiliary_loss
                if model.user_encoder.auxiliary_loss is not None:
                    user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
                    loss += user_encoder_auxiliary_loss
                epoch_loss += float(loss) * user_ID.size(0)
                
                # 실시간 loss 표시
                train_dataloader_with_progress.set_postfix({'loss': f'{float(loss):.4f}', 'avg_loss': f'{epoch_loss / ((train_dataloader_with_progress.n + 1) * self.batch_size):.4f}'})
                
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                self.scheduler.step()  # LR scheduler step (10% warmup + linear decay)
            print('Epoch %d : train done' % e)
            print('loss =', epoch_loss / len(self.train_dataset))

            # validation
            auc, mrr, ndcg5, ndcg10 = compute_scores(model, self.mind_corpus, self.batch_size * 3 // 2, 'dev', self.dev_res_dir + '/' + model.model_name + '-' + str(e) + '.txt', self._dataset)
            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
            if self.dev_criterion == 'auc':
                if auc >= self.best_dev_auc:
                    self.best_dev_auc = auc
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'mrr':
                if mrr >= self.best_dev_mrr:
                    self.best_dev_mrr = mrr
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg5':
                if ndcg5 >= self.best_dev_ndcg5:
                    self.best_dev_ndcg5 = ndcg5
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg10':
                if ndcg10 >= self.best_dev_ndcg10:
                    self.best_dev_ndcg10 = ndcg10
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            else:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= self.best_dev_avg:
                    self.best_dev_avg = avg
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1

            print('Best epoch :', self.best_dev_epoch)
            print('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
            torch.cuda.empty_cache()
            if self.epoch_not_increase == 0:
                torch.save({model.model_name: model.state_dict()}, self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch))
            if self.epoch_not_increase == self.early_stopping_epoch:
                break

        with open('%s/%s-%s-dev_log.txt' % (self.dev_res_dir, model.model_name, self._dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(self.auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, self.auc_results[i], self.mrr_results[i], self.ndcg5_results[i], self.ndcg10_results[i]))
        shutil.copy(self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch), self.best_model_dir + '/' + model.model_name)
        print('Training : ' + model.model_name + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])


def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss

def distributed_train(rank, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
    world_size = config.world_size
    model_name = model.model_name

    # NCCL 초기화 (timeout 사실상 무한대로 설정)
    import os
    os.environ['NCCL_BLOCKING_WAIT'] = '0'  # Non-blocking wait
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Async error handling

    import datetime
    # Dev 평가 시간 예측 불가 → timeout 매우 크게 (24시간)
    timeout = datetime.timedelta(days=1)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=timeout)
    config.device_id = rank
    config.set_cuda()
    model.cuda()
    loss_ = negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else negative_log_sigmoid
    epoch = config.epoch
    batch_size = config.batch_size // world_size
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # PLM 기반 모델의 경우 learning rate 분리
    if config.use_plm_news_encoder:
        # PLM 파라미터와 나머지 파라미터 분리
        plm_params = []
        other_params = []

        for name, param in model.module.named_parameters():
            if param.requires_grad:
                if 'plm' in name:  # PLM 파라미터
                    plm_params.append(param)
                else:  # 다른 파라미터 (category embedding, attention 등)
                    other_params.append(param)

        # 서로 다른 learning rate 설정
        optimizer = optim.Adam([
            {'params': plm_params, 'lr': config.plm_lr},      # 1e-5 (PLM)
            {'params': other_params, 'lr': config.lr}         # 1e-4 (기타)
        ], weight_decay=config.weight_decay)

        if rank == 0:
            print(f'[Multi-GPU] PLM parameters: {len(plm_params)}, Other parameters: {len(other_params)}')
            print(f'[Multi-GPU] PLM lr: {config.plm_lr}, Other lr: {config.lr}')
    else:
        # 기존 방식 (non-PLM 모델)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.module.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    # LR Scheduler (10% warmup + linear decay)
    from transformers import get_linear_schedule_with_warmup
    train_dataset = MIND_Train_Dataset(mind_corpus)
    total_steps = len(train_dataset) // batch_size * epoch
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    gradient_clip_norm = config.gradient_clip_norm
    if rank == 0:
        model_dir = config.model_dir + '/#' + str(run_index)
        best_model_dir = config.best_model_dir + '/#' + str(run_index)
        dev_res_dir = config.dev_res_dir + '/#' + str(run_index)
        result_dir = config.result_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(best_model_dir):
            os.mkdir(best_model_dir)
        if not os.path.exists(dev_res_dir):
            os.mkdir(dev_res_dir)
        with open(config.config_dir + '/#' + str(run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if config.dataset == 'large':
            prediction_dir = config.prediction_dir + '/#' + str(run_index)
            os.mkdir(prediction_dir)
        dev_criterion = config.dev_criterion
        early_stopping_epoch = config.early_stopping_epoch
        auc_results = []
        mrr_results = []
        ndcg5_results = []
        ndcg10_results = []
        best_dev_epoch = 0
        best_dev_auc = 0
        best_dev_mrr = 0
        best_dev_ndcg5 = 0
        best_dev_ndcg10 = 0
        best_dev_avg = AvgMetric(0, 0, 0, 0)
        epoch_not_increase = 0
        print('Running : ' + model_name + '\t#' + str(run_index))

    for e in tqdm(range(1, epoch + 1), desc='Epoch', disable=(rank != 0)):
        train_dataset.negative_sampling(rank=rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_sampler.set_epoch(e)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=batch_size // 16, pin_memory=True, sampler=train_sampler)
        model.train()
        epoch_loss = 0
        
        # 배치 단위 진행률 표시 (rank 0만)
        if rank == 0:
            train_dataloader_with_progress = tqdm(train_dataloader, desc=f'Epoch {e}/{epoch}', leave=False)
        else:
            train_dataloader_with_progress = train_dataloader
        
        for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
            news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) in train_dataloader_with_progress:
            user_ID = user_ID.cuda(non_blocking=True)                                                                                                                       # [batch_size]
            user_category = user_category.cuda(non_blocking=True)                                                                                                           # [batch_size, max_history_num]
            user_subCategory = user_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, max_history_num]
            user_title_text = user_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
            user_title_mask = user_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, max_history_num, max_title_length]
            user_title_entity = user_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_title_length]
            user_content_text = user_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
            user_content_mask = user_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num, max_content_length]
            user_content_entity = user_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, max_history_num, max_content_length]
            user_history_mask = user_history_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, max_history_num]
            user_history_graph = user_history_graph.cuda(non_blocking=True)                                                                                                 # [batch_size, max_history_num, max_history_num]
            user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)                                                                                 # [batch_size, category_num + 1]
            user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)                                                                           # [batch_size, max_history_num]
            news_category = news_category.cuda(non_blocking=True)                                                                                                           # [batch_size, 1 + negative_sample_num]
            news_subCategory = news_subCategory.cuda(non_blocking=True)                                                                                                     # [batch_size, 1 + negative_sample_num]
            news_title_text = news_title_text.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
            news_title_mask = news_title_mask.cuda(non_blocking=True)                                                                                                       # [batch_size, 1 + negative_sample_num, max_title_length]
            news_title_entity = news_title_entity.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_title_length]
            news_content_text = news_content_text.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
            news_content_mask = news_content_mask.cuda(non_blocking=True)                                                                                                   # [batch_size, 1 + negative_sample_num, max_content_length]
            news_content_entity = news_content_entity.cuda(non_blocking=True)                                                                                               # [batch_size, 1 + negative_sample_num, max_content_length]

            logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                           news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) # [batch_size, 1 + negative_sample_num]

            loss = loss_(logits)
            if model.module.news_encoder.auxiliary_loss is not None:
                news_auxiliary_loss = model.module.news_encoder.auxiliary_loss.mean()
                loss += news_auxiliary_loss
            if model.module.user_encoder.auxiliary_loss is not None:
                user_encoder_auxiliary_loss = model.module.user_encoder.auxiliary_loss.mean()
                loss += user_encoder_auxiliary_loss
            epoch_loss += float(loss) * user_ID.size(0)
            
            # 실시간 loss 표시 (rank 0만)
            if rank == 0 and hasattr(train_dataloader_with_progress, 'set_postfix'):
                train_dataloader_with_progress.set_postfix({'loss': f'{float(loss):.4f}', 'avg_loss': f'{epoch_loss / ((train_dataloader_with_progress.n + 1) * batch_size):.4f}'})
            
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()  # LR scheduler step (10% warmup + linear decay)
        print('rank %d : Epoch %d : train done' % (rank, e))
        print('rank %d : loss = %.6f' % (rank, epoch_loss / len(train_dataset) * world_size))

        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # dev (rank 0만 수행, rank 1은 다음 epoch 준비)
        if rank == 0:
            # Dev 배치사이즈를 더 크게 (속도 향상)
            dev_batch_size = batch_size * 4  # 메모리 고려하여 4배로 설정
            auc, mrr, ndcg5, ndcg10 = compute_scores(model.module, mind_corpus, dev_batch_size, 'dev', dev_res_dir + '/' + model_name + '-' + str(e) + '.txt', config.dataset)
            auc_results.append(auc)
            mrr_results.append(mrr)
            ndcg5_results.append(ndcg5)
            ndcg10_results.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
            if dev_criterion == 'auc':
                if auc >= best_dev_auc:
                    best_dev_auc = auc
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'mrr':
                if mrr >= best_dev_mrr:
                    best_dev_mrr = mrr
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'ndcg5':
                if ndcg5 >= best_dev_ndcg5:
                    best_dev_ndcg5 = ndcg5
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            elif dev_criterion == 'ndcg10':
                if ndcg10 >= best_dev_ndcg10:
                    best_dev_ndcg10 = ndcg10
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1
            else:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= best_dev_avg:
                    best_dev_avg = avg
                    best_dev_epoch = e
                    with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    epoch_not_increase = 0
                else:
                    epoch_not_increase += 1

            print('Best epoch :', best_dev_epoch)
            if dev_criterion == 'auc':
                print('Best AUC : %.4f' % best_dev_auc)
            elif dev_criterion == 'mrr':
                print('Best MRR : %.4f' % best_dev_mrr)
            elif dev_criterion == 'ndcg5':
                print('Best nDCG@5 : %.4f' % best_dev_ndcg5)
            elif dev_criterion == 'ndcg10':
                print('Best nDCG@10 : %.4f' % best_dev_ndcg10)
            else:
                print('Best avg : ' + str(best_dev_avg))
            torch.cuda.empty_cache()
            if epoch_not_increase == 0:
                torch.save({model_name: model.module.state_dict()}, model_dir + '/' + model_name + '-' + str(best_dev_epoch))

            # Early stopping 결정을 텐서에 저장
            early_stop_signal = torch.tensor([1 if epoch_not_increase > early_stopping_epoch else 0],
                                            dtype=torch.int32, device='cuda')
        else:
            # Rank 1은 빈 텐서 생성
            early_stop_signal = torch.tensor([0], dtype=torch.int32, device='cuda')

        # 다음 epoch 시작 전 동기화 및 early stopping 체크
        dist.barrier()
        dist.broadcast(early_stop_signal, src=0)

        if early_stop_signal[0] == 1:
            if rank == 0:
                print(f'Early stopping at epoch {e}')
            break

    if rank == 0:
        with open('%s/%s-%s-dev_log.txt' % (dev_res_dir, model_name, config.dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, auc_results[i], mrr_results[i], ndcg5_results[i], ndcg10_results[i]))
        print('Training : ' + model_name + ' #' + str(run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % auc_results[best_dev_epoch - 1])
        print('MRR : %.4f' % mrr_results[best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % ndcg5_results[best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % ndcg10_results[best_dev_epoch - 1])
        shutil.copy(model_dir + '/' + model_name + '-' + str(best_dev_epoch), best_model_dir + '/' + model_name)
        os.kill(os.getpid(), signal.SIGKILL)
