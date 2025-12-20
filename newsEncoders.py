import pickle
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention
# PLM-NR
from transformers import BertModel , RobertaModel, AutoModel

class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])
    
    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]

    def load_category_embeddings_from_glove(self, category_dict, frozen=True):
        """
        Glove로 category embedding 초기화
        Args:
            category_dict: {category_name: index} 딕셔너리
            frozen: True면 학습 불가 (논문 방식), False면 trainable
        """
        from torchtext.vocab import GloVe
        import re

        # Glove 로드 (840B, 300차원)
        print("Loading Glove 840B 300d for category embeddings...")
        glove = GloVe(
            name='840B',
            dim=300,
            cache='/home/user/jaesung/newsreclib/data/glove'
        )

        category_emb_dim = self.category_embedding.weight.size(1)  # weight.size(): [category_num, category_embedding_dim]
        print(f"Category embedding dimension: {category_emb_dim}")

        # Category name 전처리 함수
        def preprocess_category(name):
            """복합어를 분리하여 단어 리스트 반환
            Examples:
                'foodanddrink' → ['food', 'drink']
                'middleeast' → ['middle', 'east']
                'northamerica' → ['north', 'america']
            """
            # 'and'를 공백으로 치환
            name = name.replace('and', ' ')
            # 중복 공백 제거 및 소문자 변환
            words = name.lower().split()
            return words

        # Category별 초기화
        success_count = 0
        for category_name, idx in category_dict.items():
            words = preprocess_category(category_name)

            # Glove 벡터 수집
            vectors = []
            for word in words:
                if word in glove.stoi:
                    vectors.append(glove.vectors[glove.stoi[word]])

            if len(vectors) > 0:
                # 평균 벡터 계산
                avg_vector = torch.stack(vectors).mean(dim=0)  # [300]

                # Dimension 조정
                if category_emb_dim == 300:
                    category_vector = avg_vector
                elif category_emb_dim < 300:
                    # Truncate
                    category_vector = avg_vector[:category_emb_dim]
                else:
                    # Zero padding
                    padding = torch.zeros(category_emb_dim - 300)
                    category_vector = torch.cat([avg_vector, padding])

                # Embedding에 할당
                self.category_embedding.weight.data[idx] = category_vector
                success_count += 1
                print(f"  ✓ '{category_name}' initialized from Glove")
            else:
                # Glove에 없는 경우 random 유지
                print(f"  ✗ '{category_name}' uses random initialization")

        # Frozen 여부 설정
        self.category_embedding.weight.requires_grad = not frozen  # False일때 glove 벡터 그대로 유지

        status = "frozen" if frozen else "trainable"
        print(f"\nCategory embedding: {success_count}/{len(category_dict)} from Glove ({status})")


    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subCategory_embedding(subCategory)                                                                           # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, self.dropout(category_representation), self.dropout(subCategory_representation)], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CNE(NewsEncoder):
    def __init__(self, config: Config):
        super(CNE, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.word_embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.news_embedding_dim = config.hidden_dim * 4 + config.category_embedding_dim + config.subCategory_embedding_dim
        # selective LSTM encoder
        self.title_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.content_lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.title_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.title_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        self.content_H = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
        self.content_M = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=True)
        # self-attention
        self.title_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        self.content_self_attention = Attention(self.hidden_dim * 2, config.attention_dim)
        # cross-attention
        self.title_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)
        self.content_cross_attention = ScaledDotProduct_CandidateAttention(self.hidden_dim * 2, self.hidden_dim * 2, config.attention_dim)

    def initialize(self):
        super().initialize()
        for parameter in self.title_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        for parameter in self.content_lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)
        nn.init.xavier_uniform_(self.title_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.title_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.title_M.bias)
        nn.init.xavier_uniform_(self.content_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.content_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.content_M.bias)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()
        self.title_cross_attention.initialize()
        self.content_cross_attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        title_mask = title_mask.view([batch_news_num, self.max_title_length])                                                                              # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_news_num, self.max_content_length])                                                                        # [batch_size * news_num, max_content_length]
        title_mask[:, 0] = 1   # To avoid empty input of LSTM
        content_mask[:, 0] = 1 # To avoid empty input of LSTM
        title_length = title_mask.sum(dim=1, keepdim=False).long()                                                                                         # [batch_size * news_num]
        content_length = content_mask.sum(dim=1, keepdim=False).long()                                                                                     # [batch_size * news_num]
        sorted_title_length, sorted_title_indices = torch.sort(title_length, descending=True)                                                              # [batch_size * news_num]
        _, desorted_title_indices = torch.sort(sorted_title_indices, descending=False)                                                                     # [batch_size * news_num]
        sorted_content_length, sorted_content_indices = torch.sort(content_length, descending=True)                                                        # [batch_size * news_num]
        _, desorted_content_indices = torch.sort(sorted_content_indices, descending=False)                                                                 # [batch_size * news_num]
        # 1. word embedding
        title = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])                 # [batch_size * news_num, max_content_length, word_embedding_dim]
        sorted_title = pack_padded_sequence(title.index_select(0, sorted_title_indices), sorted_title_length.cpu(), batch_first=True)                      # [batch_size * news_num, max_title_length, word_embedding_dim]
        sorted_content = pack_padded_sequence(content.index_select(0, sorted_content_indices), sorted_content_length.cpu(), batch_first=True)              # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. selective LSTM encoding
        sorted_title_h, (sorted_title_h_n, sorted_title_c_n) = self.title_lstm(sorted_title)
        sorted_content_h, (sorted_content_h_n, sorted_content_c_n) = self.content_lstm(sorted_content)
        sorted_title_m = torch.cat([sorted_title_c_n[0], sorted_title_c_n[1]], dim=1)                                                                      # [batch_size * news_num, hidden_dim * 2]
        sorted_content_m = torch.cat([sorted_content_c_n[0], sorted_content_c_n[1]], dim=1)                                                                # [batch_size * news_num, hidden_dim * 2]
        sorted_title_h, _ = pad_packed_sequence(sorted_title_h, batch_first=True, total_length=self.max_title_length)                                      # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_h, _ = pad_packed_sequence(sorted_content_h, batch_first=True, total_length=self.max_content_length)                                # [batch_size * news_num, max_content_length, hidden_dim * 2]
        sorted_title_gate = torch.sigmoid(self.title_H(sorted_title_h) + self.title_M(sorted_content_m).unsqueeze(dim=1))        # gate                          # [batch_size * news_num, max_title_length, hidden_dim * 2]
        sorted_content_gate = torch.sigmoid(self.content_H(sorted_content_h) + self.content_M(sorted_title_m).unsqueeze(dim=1))  # gate                          # [batch_size * news_num, max_content_length, hidden_dim * 2]
        title_h = (sorted_title_h * sorted_title_gate).index_select(0, desorted_title_indices)                                                             # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = (sorted_content_h * sorted_content_gate).index_select(0, desorted_content_indices)                                                     # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask)                                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                                # [batch_size * news_num, hidden_dim * 2]
        # 4. cross-attention
        title_cross = self.title_cross_attention(title_h, content_self, title_mask)                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_cross = self.content_cross_attention(content_h, title_self, content_mask)                                                                  # [batch_size * news_num, hidden_dim * 2]
        news_representation = torch.cat([title_self + title_cross, content_self + content_cross], dim=1).view([batch_size, news_num, self.hidden_dim * 4]) # [batch_size, news_num, hidden_dim * 4]
        # 5. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                              # [batch_size, news_num, news_embedding_dim]
        return news_representation


class CNN(NewsEncoder):
    def __init__(self, config: Config):
        super(CNN, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.cnn_kernel_num])                        # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class MHSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MHSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.feature_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(config.head_num*config.head_dim, config.attention_dim)
        self.news_embedding_dim = config.head_num * config.head_dim + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                    # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.feature_dim])                           # [batch_size, news_num, news_embedding_dim]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class KCNN(NewsEncoder):
    def __init__(self, config: Config):
        super(KCNN, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.entity_embedding_dim = config.entity_embedding_dim
        self.context_embedding_dim = config.context_embedding_dim
        self.entity_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.entity_embedding_dim)
        self.context_embedding = nn.Embedding(num_embeddings=config.entity_size, embedding_dim=self.context_embedding_dim)
        with open('entity_embedding-%s.pkl' % config.dataset, 'rb') as entity_embedding_f:
            self.entity_embedding.weight.data.copy_(pickle.load(entity_embedding_f))
        with open('context_embedding-%s.pkl' % config.dataset, 'rb') as context_embedding_f:
            self.context_embedding.weight.data.copy_(pickle.load(context_embedding_f))
        self.M_entity = nn.Linear(self.entity_embedding_dim, self.word_embedding_dim, bias=True)
        self.M_context = nn.Linear(self.context_embedding_dim, self.word_embedding_dim, bias=True)
        self.knowledge_cnn = Conv2D_Pool(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size, 3)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.M_entity.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_entity.bias)
        nn.init.xavier_uniform_(self.M_context.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.M_context.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word & entity & context embedding
        word_embedding = self.word_embedding(title_text).view([batch_news_num, self.max_title_length, self.word_embedding_dim])                                  # [batch_size * news_num, max_title_length, word_embedding_dim]
        entity_embedding = self.entity_embedding(title_entity).view([batch_news_num, self.max_title_length, self.entity_embedding_dim])                          # [batch_size * news_num, max_title_length, entity_embedding_dim]
        context_embedding = self.context_embedding(title_entity).view([batch_news_num, self.max_title_length, self.context_embedding_dim])                       # [batch_size * news_num, max_title_length, context_embedding_dim]
        W = torch.stack([word_embedding, torch.tanh(self.M_entity(entity_embedding)), torch.tanh(self.M_context(context_embedding))], dim=3).permute(0, 2, 1, 3) # [batch_size * news_num, word_embedding_dim, max_title_length, 3]
        # 2. knowledge-aware CNN
        news_representation = self.knowledge_cnn(W).view([batch_size, news_num, self.cnn_kernel_num])                                                            # [batch_size, news_num, cnn_kernel_num]
        # 3. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                                                    # [batch_size, news_num, news_embedding_dim]
        return news_representation


class HDC(NewsEncoder):
    def __init__(self, config: Config):
        super(HDC, self).__init__(config)
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.word_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.word_embedding_dim)
        self.HDC_sequence_length = config.max_title_length + 2
        self.HDC_filter_num = config.HDC_filter_num
        self.dilated_conv1 = nn.Conv1d(in_channels=config.word_embedding_dim, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2, dilation=1)
        self.dilated_conv2 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 1, dilation=2)
        self.dilated_conv3 = nn.Conv1d(in_channels=self.HDC_filter_num, out_channels=self.HDC_filter_num, kernel_size=config.HDC_window_size, padding=(config.HDC_window_size - 1) // 2 + 2, dilation=3)
        self.layer_norm1 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm2 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.layer_norm3 = nn.LayerNorm([self.HDC_filter_num, self.HDC_sequence_length])
        self.news_embedding_dim = None

    def initialize(self):
        super().initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. sequence embeddings
        word_embedding = self.word_embedding(title_text).permute(0, 1, 3, 2)                                                 # [batch_size, news_num, word_embedding_dim, title_length]
        category_embedding = self.category_embedding(category).unsqueeze(dim=3)                                              # [batch_size, news_num, word_embedding_dim, 1]
        subCategory_embedding = self.subCategory_embedding(subCategory).unsqueeze(dim=3)                                     # [batch_size, news_num, word_embedding_dim, 1]
        d0 = torch.cat([category_embedding, subCategory_embedding, word_embedding], dim=3)                                   # [batch_size, news_num, word_embedding_dim, HDC_sequence_length]
        d0 = d0.view([batch_news_num, self.word_embedding_dim, self.HDC_sequence_length])                                    # [batch_size * news_num, word_embedding_dim, HDC_sequence_length]
        # 2. hierarchical dilated convolution
        d1 = F.relu(self.layer_norm1(self.dilated_conv1(d0)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d2 = F.relu(self.layer_norm2(self.dilated_conv2(d1)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d3 = F.relu(self.layer_norm3(self.dilated_conv3(d2)), inplace=True)                                                  # [batch_size * news_num, HDC_filter_num, HDC_sequence_length]
        d0 = d0.view([batch_size, news_num, self.word_embedding_dim, self.HDC_sequence_length])                              # [batch_size, news_num, word_embedding_dim, HDC_sequence_length]
        dL = torch.stack([d1, d2, d3], dim=1).view([batch_size, news_num, 3, self.HDC_filter_num, self.HDC_sequence_length]) # [batch_size, news_num, 3, HDC_filter_num, HDC_sequence_length]
        return (d0, dL)


class NAML(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)

    def initialize(self):
        super().initialize()
        self.title_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.subCategory_affine.weight)
        nn.init.zeros_(self.subCategory_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        # 1. word embedding
        title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])       # [batch_size * news_num, max_title_length, word_embedding_dim]
        content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim]) # [batch_size * news_num, max_content_length, word_embedding_dim]
        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))                                                  # [batch_size * news_num, max_title_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                            # [batch_size * news_num, max_content_length, cnn_kernel_num]
        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])                               # [batch_size, news_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])                         # [batch_size, news_num, cnn_kernel_num]
        # 4. category and subCategory encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                              # [batch_size, news_num, cnn_kernel_num]
        subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)                  # [batch_size, news_num, cnn_kernel_num]
        # 5. multi-view attention
        feature = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)    # [batch_size, news_num, 4, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                            # [batch_size, news_num, 4, 1]
        news_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                    # [batch_size, news_num, cnn_kernel_num]
        return news_representation


class PNE(NewsEncoder):
    def __init__(self, config: Config):
        super(PNE, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.personalized_embedding_dim = config.personalized_embedding_dim
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.dense = nn.Linear(config.user_embedding_dim, config.personalized_embedding_dim, bias=True)
        self.personalizedAttention = CandidateAttention(config.cnn_kernel_num, config.personalized_embedding_dim, config.attention_dim)
        self.news_embedding_dim = config.cnn_kernel_num + config.category_embedding_dim + config.subCategory_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.dense.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.dense.bias)
        self.personalizedAttention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        c = self.dropout_(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                           # [batch_size * news_num, max_sentence_length, cnn_kernel_num]
        # 3. attention layer
        q_w = F.relu(self.dense(user_embedding), inplace=True).repeat([news_num, 1])                                                # [batch_size * news_num, personalized_embedding_dim]
        news_representation = self.personalizedAttention(c, q_w, mask).view([batch_size, news_num, self.cnn_kernel_num])            # [batch_size, news_num, cnn_kernel_num]
        # 4. feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                                       # [batch_size, news_num, news_embedding_dim]
        return news_representation


class DAE(NewsEncoder):
    def __init__(self, config: Config):
        super(DAE, self).__init__(config)
        self.Alpha = config.Alpha
        assert self.Alpha > 0, 'Reconstruction loss weight must be greater than 0'
        self.f1 = nn.Linear(config.word_embedding_dim, config.hidden_dim, bias=True)
        self.f2 = nn.Linear(config.hidden_dim, config.word_embedding_dim, bias=True)
        self.news_embedding_dim = config.hidden_dim + config.category_embedding_dim + config.subCategory_embedding_dim
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.f1.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f1.bias)
        nn.init.xavier_uniform_(self.f2.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.f2.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        title_mask = title_mask.unsqueeze(dim=3)
        content_mask = content_mask.unsqueeze(dim=3)
        word_embedding = torch.sigmoid(((self.word_embedding(title_text) * title_mask).sum(dim=2) + (self.word_embedding(content_text) * content_mask).sum(dim=2)) \
                         / (title_mask.sum(dim=2, keepdim=False) + content_mask.sum(dim=2, keepdim=False)))           # [batch_size, news_num, word_embedding_dim]
        corrupted_word_embedding = self.dropout_(word_embedding)                                                      # [batch_size, news_num, word_embedding_dim]
        news_representation = torch.sigmoid(self.f1(corrupted_word_embedding))                                        # [batch_size, news_num, news_embedding_dim]
        denoised_word_embedding = torch.sigmoid(self.f2(news_representation))                                         # [batch_size, news_num, word_embedding_dim]
        self.auxiliary_loss = torch.norm(word_embedding - denoised_word_embedding, dim=2, keepdim=False) * self.Alpha # [batch_size, news_num]
        # feature fusion
        news_representation = self.feature_fusion(news_representation, category, subCategory)                         # [batch_size, news_num, news_embedding_dim]
        return news_representation


class Inception(NewsEncoder):
    def __init__(self, config: Config):
        super(Inception, self).__init__(config)
        assert config.word_embedding_dim == config.category_embedding_dim and config.word_embedding_dim == config.subCategory_embedding_dim, 'embedding dimension must be the same in the Inception module'
        self.fc1_1 = nn.Linear(config.word_embedding_dim*4, config.hidden_dim, bias=True)
        self.fc1_2 = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.fc1_3 = nn.Linear(config.hidden_dim, config.word_embedding_dim, bias=True)
        self.fc2 = nn.Linear(config.word_embedding_dim*4, config.word_embedding_dim, bias=True)
        self.linear_transform = nn.Linear(config.word_embedding_dim*3, config.word_embedding_dim, bias=True)
        self.news_embedding_dim = config.word_embedding_dim

    def initialize(self):
        super().initialize()
        nn.init.xavier_uniform_(self.fc1_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1_1.bias)
        nn.init.xavier_uniform_(self.fc1_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1_2.bias)
        nn.init.xavier_uniform_(self.fc1_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1_3.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.linear_transform.weight)
        nn.init.zeros_(self.linear_transform.bias)

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        title_mask[:, :, 0] = 1   # To avoid zero-length title
        content_mask[:, :, 0] = 1 # To avoid zero-length content
        title_embedding = (self.word_embedding(title_text) * title_mask.unsqueeze(dim=3)).sum(dim=2) / title_mask.sum(dim=2, keepdim=True)         # [batch_size, news_num, word_embedding_dim]
        content_embedding = (self.word_embedding(content_text) * content_mask.unsqueeze(dim=3)).sum(dim=2) / content_mask.sum(dim=2, keepdim=True) # [batch_size, news_num, word_embedding_dim]
        category_embedding = self.category_embedding(category)                                                                                     # [batch_size, news_num, category_embedding_dim]
        subCategory_embedding = self.subCategory_embedding(subCategory)                                                                            # [batch_size, news_num, subCategory_embedding_dim]
        embeddings = torch.cat([title_embedding, content_embedding, category_embedding, subCategory_embedding], dim=2)                             # [batch_size, news_num, embedding_dim * 4]
        subnetwork1 = F.relu(self.fc1_3(F.relu(self.fc1_2(F.relu(self.fc1_1(embeddings), inplace=True)), inplace=True)), inplace=True)             # [batch_size, news_num, embedding_dim]
        subnetwork2 = F.relu(self.fc2(embeddings), inplace=True)                                                                                   # [batch_size, news_num, embedding_dim]
        subnetwork3 = title_embedding + content_embedding + category_embedding + subCategory_embedding                                             # [batch_size, news_num, embedding_dim]
        news_representation = self.linear_transform(torch.cat([subnetwork1, subnetwork2, subnetwork3], dim=2))                                     # [batch_size, news_num, embedding_dim]
        return news_representation


class PLMNewsEncoder(NewsEncoder):
    def __init__(self, config: Config):
        super(PLMNewsEncoder, self).__init__(config)
        self.config = config

        # 1. PLM 로딩
        self.plm_hidden_dim = 768 # Bert-Base hidden size
        if config.plm_type == 'bert':
            self.plm = BertModel.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')
            self.plm_hidden_dim = self.plm.config.hidden_size
        elif config.plm_type == 'roberta':
            self.plm = RobertaModel.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')
            self.plm_hidden_dim = self.plm.config.hidden_size
        else:
            self.plm = AutoModel.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')
            self.plm_hidden_dim = self.plm.config.hidden_size

        # 2. PLM Layer Freezing (마지막 N개 레이어만 학습)
        self._freeze_plm_layers(config.plm_frozen_layers)

        # 3. Pooling -> attention
        self.pooling_method = config.plm_pooling
        if self.pooling_method == 'attention':
            self.attention = Attention(feature_dim=self.plm_hidden_dim, attention_dim=config.attention_dim)
        
        # 4. Dropout 설정
        self.dropout =  nn.Dropout(p=config.dropout_rate, inplace=False)

        # 5. Auxiliary loss
        self.auxiliary_loss = None
    
    def _freeze_plm_layers(self, frozen_layers):
        """PLM의 하위 레이어 frozen"""
        if frozen_layers == 0:
            # 전체 fine-tune
            return

        # BERT/RoBERTa 구조: encoder.layer.0 ~ encoder.layer.11 총 12개층으로 구성됨
        total_layers = len(self.plm.encoder.layer)

        for layer_idx in range(frozen_layers):
            for param in self.plm.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

        print(f'Frozen {frozen_layers} layers out of {total_layers} in PLM')

    def _pool_hidden_states(self, hidden_states, attention_mask):
        """
        PLM hidden states를 news embedding으로 pooling

        Args:
            hidden_states: [batch_news_num, seq_len, hidden_dim]
            attention_mask: [batch_news_num, seq_len]

        Returns:
            news_embedding: [batch_news_num, hidden_dim]
        """
        if self.pooling_method == 'cls':
            # [CLS] 토큰의 representation 사용, hidden_states: [batch_news_num, seq_len, hidden_dim] -> [B*N, hidden_dim]
            news_embedding = hidden_states[:, 0, :]  # [B*N, hidden_dim]

        elif self.pooling_method == 'average':
            # Attention mask를 고려한 평균
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)  # [B*N, hidden_dim]
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            news_embedding = sum_hidden / sum_mask

        elif self.pooling_method == 'attention':
            # Attention 메커니즘
            news_embedding = self.attention(hidden_states, mask=attention_mask)  # [B*N, hidden_dim]

        else:
            raise ValueError(f'Unknown pooling method: {self.pooling_method}')

        return news_embedding

    def initialize(self):
        """파라미터 초기화"""
        if self.pooling_method == 'attention':
            self.attention.initialize()

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise NotImplementedError('Subclass must implement forward()')


class PLMNAML(PLMNewsEncoder):
    """
    PLM-empowered NAML
    - PLM for title encoding
    - Category/SubCategory fusion
    """

    def __init__(self, config: Config):
        super(PLMNAML, self).__init__(config)

        # Category embeddings
        self.category_embedding = nn.Embedding(
            num_embeddings=config.category_num,
            embedding_dim=config.category_embedding_dim
        )
        self.subCategory_embedding = nn.Embedding(
            num_embeddings=config.subCategory_num,
            embedding_dim=config.subCategory_embedding_dim
        )

        # News embedding dimension
        self.news_embedding_dim = (
            self.plm_hidden_dim +
            config.category_embedding_dim +
            config.subCategory_embedding_dim
        )

    def initialize(self):
        super().initialize()
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        """
        Args:
            title_text: [batch_size, news_num, max_title_length] - PLM token IDs
            title_mask: [batch_size, news_num, max_title_length] - PLM attention mask
            category: [batch_size, news_num]
            subCategory: [batch_size, news_num]

        Returns:
            news_representation: [batch_size, news_num, news_embedding_dim]
        """
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        max_title_length = title_text.size(2)
        batch_news_num = batch_size * news_num

        # 1. Reshape for PLM
        title_text = title_text.view([batch_news_num, max_title_length])
        title_mask = title_mask.view([batch_news_num, max_title_length])

        # 2. PLM encoding
        plm_output = self.plm(
            input_ids=title_text,
            attention_mask=title_mask,
            return_dict=True
        )
        hidden_states = plm_output.last_hidden_state  # [B*N, seq_len, 768]

        # 3. Pooling
        news_repr = self._pool_hidden_states(hidden_states, title_mask)  # [B*N, 768]
        news_repr = self.dropout(news_repr)
        news_repr = news_repr.view([batch_size, news_num, self.plm_hidden_dim])

        # 4. Feature fusion (category/subcategory concatenation)
        news_representation = self.feature_fusion(news_repr, category, subCategory)  # [B, N, 768+50+50=868]

        return news_representation


class PLMNRMS(PLMNewsEncoder):
    """
    PLM-empowered NRMS
    - PLM only (no category fusion)
    """

    def __init__(self, config: Config):
        super(PLMNRMS, self).__init__(config)
        self.news_embedding_dim = self.plm_hidden_dim

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        max_title_length = title_text.size(2)
        batch_news_num = batch_size * news_num

        # 1. Reshape
        title_text = title_text.view([batch_news_num, max_title_length])
        title_mask = title_mask.view([batch_news_num, max_title_length])

        # 2. PLM encoding
        plm_output = self.plm(
            input_ids=title_text,
            attention_mask=title_mask,
            return_dict=True
        )
        hidden_states = plm_output.last_hidden_state

        # 3. Pooling
        news_repr = self._pool_hidden_states(hidden_states, title_mask)
        news_repr = self.dropout(news_repr)
        news_representation = news_repr.view([batch_size, news_num, self.plm_hidden_dim])

        return news_representation


class PLMMiner(PLMNewsEncoder):
    """
    PLM-empowered encoder for MINER user encoder
    - PLM for title encoding
    - Category/SubCategory fusion
    - Automatic GloVe initialization for category embeddings (for category-aware attention)
    """

    def __init__(self, config: Config, category_dict: dict):
        super(PLMMiner, self).__init__(config)

        # Category embeddings
        self.category_embedding = nn.Embedding(
            num_embeddings=config.category_num,
            embedding_dim=config.category_embedding_dim
        )

        # News embedding dimension
        self.news_embedding_dim = self.plm_hidden_dim

        # Store category_dict for GloVe initialization
        self.category_dict = category_dict
        self.use_category_glove = config.use_category_glove

    def initialize(self):
        super().initialize()

        # Default: random uniform initialization
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

        # GloVe initialization for category embeddings (if enabled)
        if self.use_category_glove:
            print("\n" + "="*60)
            print("PLMMiner: Initializing category embeddings with GloVe 840B 300d")
            print("="*60)
            self.load_category_embeddings_from_glove(
                category_dict=self.category_dict,
                frozen=True  # 카테고리 임베딩은 훈련하면 안됨
            )
            print("="*60 + "\n")

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        """
        Args:
            title_text: [batch_size, news_num, max_title_length] - PLM token IDs
            title_mask: [batch_size, news_num, max_title_length] - PLM attention mask
            category: [batch_size, news_num]
            subCategory: [batch_size, news_num]

        Returns:
            news_representation: [batch_size, news_num, news_embedding_dim]
        """
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        max_title_length = title_text.size(2)
        batch_news_num = batch_size * news_num

        # 1. Reshape for PLM
        title_text = title_text.view([batch_news_num, max_title_length])
        title_mask = title_mask.view([batch_news_num, max_title_length])

        # 2. PLM encoding
        plm_output = self.plm(
            input_ids=title_text,
            attention_mask=title_mask,
            return_dict=True
        )
        hidden_states = plm_output.last_hidden_state  # [B*N, seq_len, 768]

        # 3. Pooling
        news_repr = self._pool_hidden_states(hidden_states, title_mask)  # [B*N, 768]
        news_repr = self.dropout(news_repr)
        news_repr = news_repr.view([batch_size, news_num, self.plm_hidden_dim])

        return news_repr


