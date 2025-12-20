# PLMMiner 모델 통합 가이드

**날짜**: 2025-12-20
**모델**: PLMMiner (News Encoder) + MINER (User Encoder)
**데이터셋**: MIND (Microsoft News Dataset)

---

## 📋 목차

1. [실행 명령어](#1-실행-명령어)
2. [Git 변경사항 요약](#2-git-변경사항-요약)
3. [모델 아키텍처](#3-모델-아키텍처)

---

## 1. 실행 명령어

### 기본 훈련

```bash
python main.py \
    --news_encoder=PLMMiner \
    --user_encoder=MINER \
    --dataset=small \
    --mode=train \
    --use_category_glove \
    --batch_size=32 \
    --epoch=5
```

### 주요 파라미터

| 파라미터 | 설명 | 기본값 | 권장값 |
|---------|------|--------|--------|
| `--news_encoder` | 뉴스 인코더 선택 | - | `PLMMiner` |
| `--user_encoder` | 유저 인코더 선택 | - | `MINER` |
| `--dataset` | 데이터셋 크기 | `small` | `small/200k/large` |
| `--use_category_glove` | GloVe 카테고리 초기화 | False | 사용 권장 |
| `--plm_type` | PLM 모델 타입 | `bert` | `bert/roberta` |
| `--plm_model_name` | PLM 모델 이름 | `bert-base-uncased` | - |
| `--plm_frozen_layers` | PLM 동결 레이어 수 | 10 | 10 (상위 2개층만 학습) |
| `--plm_lr` | PLM learning rate | 1e-5 | 1e-5 |
| `--num_interest_vectors` | Interest vector 개수 (K) | 32 | 32 |
| `--context_code_dim` | Context code 차원 | 200 | 200 |
| `--category_aware_lambda` | Category similarity 가중치 | 0.5 | 0.5 |
| `--miner_aggregation` | Score 집계 방식 | `weighted` | `weighted/max/mean` |
| `--disagreement_beta` | Disagreement loss 가중치 | 0.8 | 0.8 |

### Dev/Test 평가

```bash
# Dev 평가
python main.py \
    --mode=dev \
    --news_encoder=PLMMiner \
    --user_encoder=MINER \
    --dataset=small \
    --dev_model_path=best_model/small/PLMMiner-MINER/#2/PLMMiner-MINER

# Test 평가
python main.py \
    --mode=test \
    --news_encoder=PLMMiner \
    --user_encoder=MINER \
    --dataset=small \
    --test_model_path=best_model/small/PLMMiner-MINER/#2/PLMMiner-MINER
```

---

## 2. Git 변경사항 요약

### 파일별 변경 내용

#### 2.1 MIND_corpus.py

**위치**: Line 482-486

**변경 전:**
```python
                self.news_plm_title_ids = data['title_ids']
                self.news_plm_title_masks = data['title_masks']

        print(f'PLM tokenization completed. Shape: {self.news_plm_title_ids.shape}')
```

**변경 후:**
```python
                self.news_plm_title_ids = data['title_ids']
                self.news_plm_title_masks = data['title_masks']

        print(f'PLM preprocessing completed. Replacing news_title_text with PLM tokenized data.')
        # PLM 데이터를 기본 데이터로 교체 (기존 데이터셋 클래스와 호환성 유지)
        self.news_title_text = self.news_plm_title_ids
        self.news_title_mask = self.news_plm_title_masks

        print(f'PLM tokenization completed. Shape: {self.news_plm_title_ids.shape}')
```

**💡 핵심 요약:**
- PLM 토큰화 데이터를 기존 필드(`news_title_text`, `news_title_mask`)에 오버라이드하여 데이터 로더 호환성 유지
- 기존 Word2Vec 기반 데이터 구조를 재사용하면서 PLM 데이터를 투명하게 통합

---

#### 2.2 config.py

**위치**: Line 15-16

**변경 전:**
```python
parser.add_argument('--news_encoder', type=str, default='CNE',
    choices=['CNE', 'CNN', 'MHSA', ..., 'NAML_Title', ...])
parser.add_argument('--user_encoder', type=str, default='SUE',
    choices=['SUE', 'LSTUR', 'MHSA', ..., 'SUE_wo_HCA'])
```

**변경 후:**
```python
parser.add_argument('--news_encoder', type=str, default='CNE',
    choices=['CNE', 'CNN', ..., 'PLMNAML', 'PLMNRMS', 'PLMMiner', ...])
parser.add_argument('--user_encoder', type=str, default='SUE',
    choices=['SUE', 'LSTUR', ..., 'MINER', 'SUE_wo_HCA'])
```

**💡 핵심 요약:**
- `PLMMiner`, `MINER` 인코더 선택 옵션 추가

---

**위치**: Line 84-92 (MINER 하이퍼파라미터 추가)

**변경 전:**
```python
parser.add_argument('--use_plm_news_encoder', action='store_true')

        self.attribute_dict = dict(vars(parser.parse_args()))
```

**변경 후:**
```python
parser.add_argument('--use_plm_news_encoder', action='store_true')
        # MINER-specific parameters
        parser.add_argument('--num_interest_vectors', type=int, default=32)
        parser.add_argument('--context_code_dim', type=int, default=200)
        parser.add_argument('--disagreement_beta', type=float, default=0.8)
        parser.add_argument('--miner_aggregation', type=str, default='weighted', choices=['max', 'mean', 'weighted'])
        parser.add_argument('--category_aware_lambda', type=float, default=0.5)
        parser.add_argument('--use_category_glove', action='store_true')

        self.attribute_dict = dict(vars(parser.parse_args()))

        # PLM 기반 뉴스 인코더 자동 설정
        if self.news_encoder in ['PLMNAML', 'PLMNRMS', 'PLMMiner']:
            self.use_plm_news_encoder = True
```

**💡 핵심 요약:**
- **MINER 핵심 파라미터 6개 추가**: Interest vectors 개수(K=32), Context dimension(200), Disagreement loss(β=0.8), Aggregation 방식(weighted), Category-aware lambda(λ=0.5), GloVe 초기화 플래그
- **자동 설정 로직**: PLM 기반 인코더 선택 시 `use_plm_news_encoder` 자동 활성화로 중복 플래그 불필요

---

#### 2.3 main.py

**위치**: Line 13-15, 38, 52

**변경 전:**
```python
model = Model(config)
```

**변경 후:**
```python
model = Model(config, mind_corpus.category_dict)
```

**💡 핵심 요약:**
- PLMMiner가 GloVe로 카테고리 임베딩 초기화할 때 필요한 `category_dict` (카테고리 이름→ID 매핑) 전달
- 카테고리 이름 (예: 'sports', 'entertainment')을 GloVe 벡터로 변환하기 위한 메타데이터 제공

---

#### 2.4 model.py

**위치**: Line 10-28

**변경 전:**
```python
class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.use_plm_news_encoder:
            if config.new_encoder == 'NAML':
                self.news_encoder = newsEncoders.PLMNAML(config)
            elif config.news_encoder == 'NRMS':
                self.news_encoder = newsEncoders.PLMNRMS(config)
        if config.news_encoder == 'CNE':
```

**변경 후:**
```python
class Model(nn.Module):
    def __init__(self, config: Config, category_dict: dict = None):
        super(Model, self).__init__()

        if config.news_encoder == 'PLMNAML':
            self.news_encoder = newsEncoders.PLMNAML(config)
        elif config.news_encoder == 'PLMNRMS':
            self.news_encoder = newsEncoders.PLMNRMS(config)
        elif config.news_encoder == 'PLMMiner':
            assert category_dict is not None, 'PLMMiner requires category_dict'
            self.news_encoder = newsEncoders.PLMMiner(config, category_dict)
        elif config.news_encoder == 'CNE':
```

**💡 핵심 요약:**
- **분기 구조 개선**: `use_plm_news_encoder` 플래그 기반 if문을 명시적 인코더명 기반 if/elif로 변경하여 코드 명확성 향상
- **PLMMiner 추가**: `category_dict` assertion으로 GloVe 초기화에 필요한 데이터 보장

---

**위치**: Line 75-76 (MINER 유저 인코더 추가)

**변경 전:**
```python
        elif config.user_encoder == 'OMAP':
            self.user_encoder = userEncoders.OMAP(self.news_encoder, config)
        # For ablations
```

**변경 후:**
```python
        elif config.user_encoder == 'OMAP':
            self.user_encoder = userEncoders.OMAP(self.news_encoder, config)
        elif config.user_encoder == 'MINER':
            self.user_encoder = userEncoders.MINER(self.news_encoder, config)
        # For ablations
```

**💡 핵심 요약:**
- MINER 유저 인코더를 모델 선택 분기에 등록

---

**위치**: Line 134 (Forward 시그니처 변경)

**변경 전:**
```python
user_representation = self.user_encoder(..., news_representation)
```

**변경 후:**
```python
user_representation = self.user_encoder(..., news_representation, news_category)
```

**💡 핵심 요약:**
- MINER의 **category-aware attention**을 위해 후보 뉴스의 카테고리 정보 (`news_category`) 전달
- 카테고리 코사인 유사도 계산에 필요 (히스토리 뉴스 카테고리와 후보 뉴스 카테고리 간 유사도)

---

#### 2.5 newsEncoders.py

**위치**: Line 42-115 (GloVe 초기화 메서드 추가)

**변경 전:**
NewsEncoder 클래스에 해당 메서드 없음

**변경 후:**
```python
def load_category_embeddings_from_glove(self, category_dict, frozen=True):
    """GloVe 840B 300d로 category embedding 초기화"""
    glove = GloVe(name='840B', dim=300, cache='/home/user/jaesung/newsreclib/data/glove')

    category_emb_dim = self.category_embedding.weight.size(1)  # 50

    for category_name, idx in category_dict.items():
        # 복합어 처리: 'foodanddrink' → ['food', 'drink']
        words = preprocess_category(category_name)

        # GloVe 벡터 수집 및 평균
        vectors = [glove.vectors[glove.stoi[word]] for word in words if word in glove.stoi]
        if len(vectors) > 0:
            avg_vector = torch.stack(vectors).mean(dim=0)  # [300]

            # Dimension 조정 (300 → 50: Truncate)
            category_vector = avg_vector[:category_emb_dim]
            self.category_embedding.weight.data[idx] = category_vector

    # Frozen 설정
    self.category_embedding.weight.requires_grad = not frozen
```

**💡 핵심 요약:**
- **GloVe 840B 300d 활용**: 카테고리 이름의 의미적 표현을 사전학습 벡터로 초기화
- **복합어 처리**: 'foodanddrink' → ['food', 'drink']로 분리 후 평균 벡터 계산
- **차원 조정**: GloVe 300차원 → Category embedding 50차원으로 Truncate
- **Frozen 옵션**: `frozen=True`시 카테고리 임베딩 학습 비활성화 (의미적 표현 유지)
- **목적**: MINER의 category-aware attention에서 의미 있는 코사인 유사도 계산 가능

---

**위치**: Line 513-534 (PLMNewsEncoder 버그 수정)

**변경 전:**
```python
class PLMNewsEncoder(NewsEncoder):
    def __init__(self, config: Config):
        super(PLMNewsEncoder, self).__init__()  # ❌ config 누락

        if self.pooling_method == 'attention':
            self.attention = Attention(hidden_dim=self.plm_hidden_dim, ...)  # ❌ 잘못된 파라미터명
```

**변경 후:**
```python
class PLMNewsEncoder(NewsEncoder):
    def __init__(self, config: Config):
        super(PLMNewsEncoder, self).__init__(config)  # ✅ config 전달

        if self.pooling_method == 'attention':
            self.attention = Attention(feature_dim=self.plm_hidden_dim, ...)  # ✅ 올바른 파라미터명
```

**💡 핵심 요약:**
- **`super().__init__(config)` 누락 버그 수정**: NewsEncoder 초기화 시 config 전달 누락으로 인한 에러 방지
- **Attention 파라미터명 수정**: `hidden_dim` → `feature_dim`으로 Attention 클래스 인터페이스에 맞게 수정

---

**위치**: Line 703-784 (PLMMiner 클래스 추가)

**변경 전:**
파일 끝

**변경 후:**
```python
class PLMMiner(PLMNewsEncoder):
    """PLM + GloVe category embedding for MINER"""

    def __init__(self, config: Config, category_dict: dict):
        super(PLMMiner, self).__init__(config)

        # Category embeddings (MINER의 category-aware attention용)
        self.category_embedding = nn.Embedding(
            num_embeddings=config.category_num,  # 17
            embedding_dim=config.category_embedding_dim  # 50
        )

        self.news_embedding_dim = self.plm_hidden_dim  # 768
        self.category_dict = category_dict
        self.use_category_glove = config.use_category_glove

    def initialize(self):
        super().initialize()

        # Random 초기화
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

        # GloVe 초기화 (if enabled)
        if self.use_category_glove:
            self.load_category_embeddings_from_glove(
                category_dict=self.category_dict,
                frozen=True  # 카테고리 임베딩 학습 비활성화
            )

    def forward(self, title_text, title_mask, ...):
        """
        Args:
            title_text: [B, N, 32] - PLM token IDs
            title_mask: [B, N, 32] - PLM attention mask

        Returns:
            news_repr: [B, N, 768] - PLM 출력 (category/subcategory fusion 없음)
        """
        B, N, L = title_text.size()

        # Reshape: [B, N, 32] → [B*N, 32]
        title_text = title_text.view(B*N, L)
        title_mask = title_mask.view(B*N, L)

        # PLM encoding
        plm_output = self.plm(input_ids=title_text, attention_mask=title_mask)
        hidden_states = plm_output.last_hidden_state  # [B*N, 32, 768]

        # Pooling (attention-based)
        news_repr = self._pool_hidden_states(hidden_states, title_mask)  # [B*N, 768]
        news_repr = self.dropout(news_repr)

        # Reshape back: [B*N, 768] → [B, N, 768]
        news_repr = news_repr.view(B, N, 768)

        return news_repr
```

**💡 핵심 요약:**
- **PLMNAML과의 차이점**:
  - PLMNAML: PLM 출력 + Category/SubCategory fusion → `[B, N, 868]` (768+50+50)
  - PLMMiner: PLM 출력만 반환 → `[B, N, 768]`
- **Category embedding 별도 관리**: MINER가 직접 category embedding에 접근하여 category-aware attention 수행
- **GloVe 자동 초기화**: `--use_category_glove` 플래그 시 `initialize()`에서 자동으로 GloVe 초기화
- **Frozen embedding**: 카테고리 임베딩 학습 비활성화로 GloVe의 의미적 표현 보존

---

#### 2.6 trainer.py

**위치**: Line 9-36 (LR Scheduler 추가)

**변경 전:**
```python
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, model, config, mind_corpus, run_index):
        self.optimizer = optim.Adam(...)
        self.train_dataset = MIND_Train_Dataset(mind_corpus)
```

**변경 후:**
```python
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup

class Trainer:
    def __init__(self, model, config, mind_corpus, run_index):
        self.optimizer = optim.Adam(...)
        self.train_dataset = MIND_Train_Dataset(mind_corpus)

        # LR Scheduler 추가 (10% warmup + linear decay)
        total_steps = len(self.train_dataset) // config.batch_size * config.epoch
        warmup_steps = int(total_steps * 0.1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
```

**💡 핵심 요약:**
- **Learning Rate Scheduling**: 초기 10% step 동안 linear warmup → 이후 linear decay
- **PLM Fine-tuning 안정화**: Warmup으로 큰 gradient에 의한 사전학습 지식 손상 방지
- **Step 단위 업데이트**: `scheduler.step()`을 매 배치마다 호출

---

**위치**: Line 86-141 (Progress Bar 개선)

**변경 전:**
```python
for e in tqdm(range(1, self.epoch + 1)):
    for (...) in train_dataloader:
        ...
        self.optimizer.step()
```

**변경 후:**
```python
for e in tqdm(range(1, self.epoch + 1), desc='Epoch'):
    train_dataloader_with_progress = tqdm(train_dataloader, desc=f'Epoch {e}/{self.epoch}', leave=False)

    for (...) in train_dataloader_with_progress:
        ...
        epoch_loss += float(loss) * user_ID.size(0)

        # 실시간 loss 표시
        train_dataloader_with_progress.set_postfix({
            'loss': f'{float(loss):.4f}',
            'avg_loss': f'{epoch_loss / ((train_dataloader_with_progress.n + 1) * self.batch_size):.4f}'
        })

        self.optimizer.step()
        self.scheduler.step()  # LR scheduler 업데이트
```

**💡 핵심 요약:**
- **2단계 Progress Bar**: Epoch 레벨 + Batch 레벨 진행률 표시
- **실시간 Loss 모니터링**: 현재 배치 loss + 누적 평균 loss 표시로 학습 안정성 즉시 확인
- **Scheduler 통합**: 매 배치마다 `scheduler.step()` 호출

---

#### 2.7 userEncoders.py

**위치**: Line 9-10 (UserEncoder 시그니처 변경)

**변경 전:**
```python
def forward(self, ..., candidate_news_representation):
```

**변경 후:**
```antml:parameter>
def forward(self, ..., candidate_news_representation, candidate_category=None):
```

**💡 핵심 요약:**
- `candidate_category` 파라미터 추가로 MINER의 category-aware attention 지원
- 기본값 `None`으로 backward compatibility 유지 (기존 인코더는 무시)

---

**위치**: Line 375-631 (MINER 클래스 추가)

**변경 전:**
파일 끝

**변경 후:**
```python
class MINER(UserEncoder):
    """
    Multi-Interest News Encoder with Poly Attention
    - Poly Attention: K개의 context codes로 다양한 관심사 추출
    - Disagreement Regularization: Interest vector 다양성 유도
    - Category-Aware Attention: 카테고리 유사도 기반 attention 가중치 조정
    - Score Aggregation: max/mean/weighted 집계
    """

    def __init__(self, news_encoder, config):
        super(MINER, self).__init__(news_encoder, config)

        # Poly attention parameters
        self.K = config.num_interest_vectors  # 32
        self.context_dim = config.context_code_dim  # 200
        self.aggregation = config.miner_aggregation  # 'weighted'
        self.disagreement_beta = config.disagreement_beta  # 0.8

        # K개의 learnable context codes [32, 200]
        self.context_codes = nn.Parameter(torch.zeros(self.K, self.context_dim))

        # Projection layer: [768] → [200]
        self.W_h = nn.Linear(self.news_embedding_dim, self.context_dim, bias=False)

        # Target-aware attention (weighted aggregation)
        if self.aggregation == 'weighted':
            self.W_e = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)

        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.attention_scalar = math.sqrt(float(self.context_dim))  # √200

        # Category-aware attention
        self.category_aware_lambda = nn.Parameter(torch.tensor(config.category_aware_lambda))  # 0.5 (learnable)
        self.category_embedding = news_encoder.category_embedding  # PLMMiner의 category embedding 참조

    def initialize(self):
        nn.init.orthogonal_(self.context_codes.data)  # Context codes: orthogonal
        nn.init.xavier_uniform_(self.W_h.weight)  # Projection: Xavier
        if self.aggregation == 'weighted':
            nn.init.xavier_uniform_(self.W_e.weight)
            nn.init.zeros_(self.W_e.bias)
```

**💡 핵심 요약:**
- **Context Codes `[32, 200]`**: Orthogonal 초기화로 서로 직교하는 32개의 쿼리 벡터 생성 (다양한 관심사 표현)
- **Projection Layer `W_h`**: 뉴스 임베딩 768차원 → Context 차원 200차원으로 투영
- **Target-Aware Attention `W_e`**: 후보 뉴스를 고려한 관심사 가중 집계 (weighted aggregation 시)
- **Category-Aware Lambda**: Learnable 파라미터로 카테고리 유사도 가중치 조정 (초기값 0.5)
- **Category Embedding 참조**: PLMMiner의 GloVe 초기화된 category embedding 공유

---

**MINER Forward Pass 구조:**

```python
def forward(self, ..., candidate_news_representation, candidate_category=None):
    """
    Args:
        user_title_text: [B, M, 32] - 히스토리 뉴스 PLM token IDs
        user_category: [B, M] - 히스토리 뉴스 카테고리 IDs
        user_history_mask: [B, M] - 히스토리 마스크
        candidate_news_representation: [B, N, 768] - 후보 뉴스 PLM 출력
        candidate_category: [B, N] - 후보 뉴스 카테고리 IDs

    Returns:
        user_representation: [B, N, 768]
    """
    B = user_title_text.size(0)
    N = candidate_news_representation.size(1)

    # 1. 히스토리 뉴스 인코딩 [B, M, 768]
    history_embedding = self.news_encoder(user_title_text, ...)

    # 2. Poly Attention: K개의 interest vectors 추출
    interest_vectors = self.poly_attention(
        history_embedding,  # [B, M, 768]
        user_history_mask,  # [B, M]
        user_category,  # [B, M]
        candidate_category  # [B, N]
    )  # → [B, N, K, 768] (category-aware) or [B, K, 768]

    # 3. Disagreement Regularization (training 시)
    if self.training:
        self.auxiliary_loss = self.disagreement_beta * self.compute_disagreement_loss(interest_vectors)

    # 4. Score Aggregation
    if self.aggregation == 'weighted':
        # Target-aware weighted sum
        W_e_h_c = F.gelu(self.W_e(candidate_news_representation))  # [B, N, 768]
        logits = torch.matmul(W_e_h_c.unsqueeze(2), interest_vectors.transpose(2, 3)).squeeze(2)  # [B, N, K]
        alpha = F.softmax(logits, dim=2)  # [B, N, K]
        user_representation = (alpha.unsqueeze(3) * interest_vectors).sum(dim=2)  # [B, N, 768]

    return user_representation
```

**💡 핵심 요약:**
- **Poly Attention**: 히스토리 뉴스로부터 K=32개의 다양한 관심사 벡터 추출
- **Category-Aware Weighting**: 히스토리 뉴스와 후보 뉴스의 카테고리 코사인 유사도를 attention에 반영
- **Disagreement Loss**: Interest vector 간 코사인 유사도 최소화로 다양성 강제 (auxiliary loss)
- **Weighted Aggregation**: 후보 뉴스별로 가장 관련 높은 interest vectors에 집중

---

**Poly Attention 세부 구조:**

```python
def poly_attention(self, history_embeddings, history_mask, user_category=None, candidate_category=None):
    """
    Args:
        history_embeddings: [B, M, 768]
        history_mask: [B, M]
        user_category: [B, M] - 히스토리 카테고리
        candidate_category: [B, N] - 후보 카테고리

    Returns:
        interest_vectors: [B, N, K, 768] (category-aware) or [B, K, 768]
    """
    B, M, D = history_embeddings.size()

    # Project history embeddings: [B, M, 768] → [B, M, 200]
    h_proj = torch.tanh(self.W_h(history_embeddings))

    # Attention logits: [B, M, 200] @ [200, 32] = [B, M, 32]
    logits = torch.matmul(h_proj, self.context_codes.T) / self.attention_scalar

    if user_category is not None and candidate_category is not None:
        # Category-aware attention
        hist_cat_emb = self.category_embedding(user_category)  # [B, M, 50]
        cand_cat_emb = self.category_embedding(candidate_category)  # [B, N, 50]

        # Cosine similarity: [B, M, 50] @ [B, 50, N] = [B, M, N]
        hist_cat_norm = F.normalize(hist_cat_emb, p=2, dim=2)
        cand_cat_norm = F.normalize(cand_cat_emb, p=2, dim=2)
        category_sim = torch.bmm(hist_cat_norm, cand_cat_norm.transpose(1, 2))

        # Expand logits: [B, M, 32] → [B, M, N, 32]
        N = candidate_category.size(1)
        logits_expanded = logits.unsqueeze(2).expand(-1, -1, N, -1)

        # Add category bias: logits + λ * cos(category)
        category_bias = self.category_aware_lambda * category_sim.unsqueeze(3)  # [B, M, N, 1]
        logits = logits_expanded + category_bias  # [B, M, N, 32]

        # Mask and softmax
        mask_expanded = history_mask.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, self.K)
        logits = logits.masked_fill(mask_expanded == 0, -1e9)
        attn_weights = F.softmax(logits, dim=1)  # [B, M, N, 32]

        # Weighted sum: [B, N, 32, M] @ [B, M, 768] = [B, N, 32, 768]
        interest_vectors = torch.einsum('bmnk,bmd->bnkd', attn_weights, history_embeddings)
    else:
        # Category-agnostic poly attention
        mask_expanded = history_mask.unsqueeze(2).expand(-1, -1, self.K)
        logits = logits.masked_fill(mask_expanded == 0, -1e9)
        attn_weights = F.softmax(logits, dim=1)  # [B, M, 32]
        interest_vectors = torch.bmm(attn_weights.transpose(1, 2), history_embeddings)  # [B, 32, 768]

    return interest_vectors
```

**💡 핵심 요약:**
- **Additive Attention**: `tanh(W_h @ h_j)` projection 후 context codes와 내적
- **Category-Aware Weighting**:
  - 히스토리 뉴스와 후보 뉴스의 카테고리 임베딩 코사인 유사도 계산
  - `λ * cos(cat_history, cat_candidate)`를 attention logits에 가산
  - 같은 카테고리 히스토리 뉴스에 더 높은 attention 부여
- **차원 확장**: [B, M, K] → [B, M, N, K]로 확장하여 후보 뉴스별 관심사 벡터 생성
- **GloVe 의존성**: `category_embedding`이 GloVe로 초기화되어야 의미 있는 코사인 유사도 계산 가능

---

**Disagreement Regularization:**

```python
def compute_disagreement_loss(self, interest_vectors):
    """
    Args:
        interest_vectors: [B, K, 768] or [B, N, K, 768]

    Returns:
        loss: scalar (평균 코사인 유사도)
    """
    # Reshape if needed
    if interest_vectors.dim() == 4:
        B, N, K, D = interest_vectors.size()
        interest_vectors = interest_vectors.view(B*N, K, D)

    # Normalize: [B, K, 768]
    normalized = F.normalize(interest_vectors, p=2, dim=2)

    # Pairwise cosine similarity: [B, K, 768] @ [B, 768, K] = [B, K, K]
    similarity_matrix = torch.bmm(normalized, normalized.transpose(1, 2))

    # Average over all pairs
    loss = similarity_matrix.sum(dim=(1, 2)) / (K * K)

    return loss.mean()
```

**💡 핵심 요약:**
- **목적**: Interest vectors 간 다양성 유도 (서로 다른 관심사 표현)
- **방법**: K×K pairwise cosine similarity 평균을 최소화
- **효과**: Context codes가 orthogonal 초기화되었지만, 학습 과정에서 collapse 방지
- **Loss 통합**: `total_loss = click_loss + β * disagreement_loss` (β=0.8)

---

### 변경 사항 전체 요약

| 파일 | 핵심 변경 | 목적 |
|------|----------|------|
| **MIND_corpus.py** | PLM 데이터를 기존 필드에 오버라이드 | 데이터 로더 호환성 유지 |
| **config.py** | MINER 파라미터 6개 추가 + PLM 자동 설정 | MINER 하이퍼파라미터 지원 |
| **main.py** | `category_dict` 전달 | GloVe 초기화용 메타데이터 제공 |
| **model.py** | PLMMiner/MINER 등록 + `news_category` 전달 | 새 인코더 지원 + category-aware attention |
| **newsEncoders.py** | GloVe 초기화 메서드 + PLMMiner 클래스 | 의미적 category embedding |
| **trainer.py** | LR Scheduler + Progress Bar | PLM fine-tuning 안정화 + 모니터링 |
| **userEncoders.py** | MINER 클래스 (260줄) | Poly attention + Category-aware attention |

---

## 3. 모델 아키텍처

### 3.1 전체 구조

```
Input (User Behavior)
  ├─ User History News: [B, M, 32] PLM token IDs
  ├─ User History Category: [B, M]
  ├─ Candidate News: [B, N, 32] PLM token IDs
  └─ Candidate Category: [B, N]

    ↓

┌─────────────────────────────────────────┐
│  PLMMiner News Encoder                  │
│  Input: [B, M+N, 32] token IDs          │
│  ├─ BERT: [B, M+N, 32, 768]             │
│  ├─ Attention Pooling: [B, M+N, 768]    │
│  └─ Output: [B, M+N, 768]               │
└─────────────────────────────────────────┘

    ↓ Split

History Embedding [B, M, 768]    Candidate Embedding [B, N, 768]

    ↓                                 ↓

┌─────────────────────────────────────────┐
│  MINER User Encoder                     │
│                                         │
│  1. Poly Attention                      │
│     ├─ Project: W_h @ h_j → [B, M, 200]│
│     ├─ Attention: logits = h_proj @ c_k │
│     ├─ Category-Aware: + λ*cos(cat)    │
│     └─ Output: [B, N, K, 768]          │
│                                         │
│  2. Disagreement Loss                   │
│     └─ Minimize cos(e_i, e_j)          │
│                                         │
│  3. Weighted Aggregation                │
│     ├─ W_e(h_c) → [B, N, 768]          │
│     ├─ Attention: softmax(W_e @ E_k)   │
│     └─ Output: [B, N, 768]             │
└─────────────────────────────────────────┘

    ↓

User Representation [B, N, 768]

    ↓

Dot Product: user_repr · candidate_repr → [B, N]

    ↓

Softmax → Click Prediction
```

---

### 3.2 PLMMiner News Encoder

#### 입력/출력 차원 변화

| Step | Operation | Input Shape | Output Shape |
|------|-----------|-------------|--------------|
| **Input** | PLM token IDs | `[B, N, 32]` | - |
| **Reshape** | `view(B*N, 32)` | `[B, N, 32]` | `[B*N, 32]` |
| **BERT** | `self.plm(input_ids, mask)` | `[B*N, 32]` | `[B*N, 32, 768]` |
| **Pooling** | Attention-based | `[B*N, 32, 768]` | `[B*N, 768]` |
| **Dropout** | - | `[B*N, 768]` | `[B*N, 768]` |
| **Reshape** | `view(B, N, 768)` | `[B*N, 768]` | `[B, N, 768]` |
| **Output** | News representation | - | `[B, N, 768]` |

**B**: Batch size (예: 64)
**N**: News 개수 (훈련: 1 positive + 4 negative = 5, 평가: 가변)
**M**: History 개수 (max 50)
**K**: Interest vectors 개수 (32)

---

#### Attention Pooling 상세

```python
def _pool_hidden_states(self, hidden_states, attention_mask):
    """
    Args:
        hidden_states: [B*N, 32, 768] - BERT 출력
        attention_mask: [B*N, 32] - Padding mask

    Returns:
        pooled: [B*N, 768]
    """
    if self.pooling_method == 'cls':
        # [CLS] 토큰 사용
        pooled = hidden_states[:, 0, :]  # [B*N, 768]

    elif self.pooling_method == 'average':
        # Masked average pooling
        mask_expanded = attention_mask.unsqueeze(2).expand_as(hidden_states)  # [B*N, 32, 768]
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # [B*N, 768]
        sum_mask = mask_expanded.sum(dim=1)  # [B*N, 768]
        pooled = sum_hidden / sum_mask.clamp(min=1e-9)  # [B*N, 768]

    elif self.pooling_method == 'attention':
        # Attention-based pooling
        attn_weights = self.attention(hidden_states, attention_mask)  # [B*N, 32, 1]
        pooled = (attn_weights * hidden_states).sum(dim=1)  # [B*N, 768]

    return pooled
```

**Attention-based Pooling 구조:**
```
hidden_states [B*N, 32, 768]
  ↓
tanh(W @ h_i) [B*N, 32, 200]
  ↓
v^T @ tanh(...) [B*N, 32, 1]
  ↓
softmax (masked) [B*N, 32, 1]
  ↓
weighted sum → [B*N, 768]
```

**💡 핵심:**
- **Attention 학습**: 각 토큰의 중요도를 학습 (제목의 핵심 단어에 집중)
- **Masking**: Padding 토큰은 softmax 전에 `-1e9`로 마스킹
- **차원 유지**: 토큰 시퀀스 `[32, 768]` → 단일 벡터 `[768]`

---

#### Category Embedding (GloVe 초기화)

```python
# 초기화 시
self.category_embedding = nn.Embedding(17, 50)  # [category_num, category_emb_dim]

# GloVe 초기화 (if --use_category_glove)
for category_name, idx in category_dict.items():
    # 'sports' → glove['sports'] [300]
    # 'foodanddrink' → (glove['food'] + glove['drink']) / 2 [300]
    glove_vector = get_glove_vector(category_name)  # [300]
    truncated_vector = glove_vector[:50]  # [50]
    self.category_embedding.weight.data[idx] = truncated_vector

self.category_embedding.weight.requires_grad = False  # Frozen
```

**GloVe 초기화 효과:**
```
Random 초기화:
  cos(sports, entertainment) = -0.03 (무의미)
  cos(sports, finance) = 0.12 (무의미)

GloVe 초기화:
  cos(sports, entertainment) = 0.42 (관련성 반영)
  cos(sports, finance) = 0.15 (낮은 관련성)
```

**💡 핵심:**
- **의미적 표현**: 카테고리 간 실제 의미적 유사도 반영
- **MINER Category-Aware Attention**: 의미 있는 코사인 유사도 계산 가능
- **Frozen**: 학습 중 GloVe의 의미적 표현 보존

---

### 3.3 MINER User Encoder

#### 전체 Forward Pass 차원 변화

| Step | Operation | Input Shape | Output Shape |
|------|-----------|-------------|--------------|
| **1. History Encoding** | PLMMiner | `[B, M, 32]` | `[B, M, 768]` |
| **2. Poly Attention** | - | - | - |
|   2-1. Projection | `W_h @ h_j` | `[B, M, 768]` | `[B, M, 200]` |
|   2-2. Logits | `h_proj @ c_k` | `[B, M, 200]` × `[32, 200]^T` | `[B, M, 32]` |
|   2-3. Category Bias | `λ * cos(cat)` | - | `[B, M, N, 1]` |
|   2-4. Expand Logits | - | `[B, M, 32]` | `[B, M, N, 32]` |
|   2-5. Attention | `softmax(logits + bias)` | `[B, M, N, 32]` | `[B, M, N, 32]` |
|   2-6. Weighted Sum | `einsum` | attn `[B, M, N, 32]` × h `[B, M, 768]` | `[B, N, 32, 768]` |
| **3. Disagreement Loss** | `cos(e_i, e_j)` | `[B, N, 32, 768]` | `scalar` |
| **4. Aggregation** | - | - | - |
|   4-1. W_e Transform | `W_e(h_c)` | `[B, N, 768]` | `[B, N, 768]` |
|   4-2. Attention | `W_e @ E_k` | `[B, N, 768]` × `[B, N, 768, 32]` | `[B, N, 32]` |
|   4-3. Softmax | - | `[B, N, 32]` | `[B, N, 32]` |
|   4-4. Weighted Sum | `α * E_k` | α `[B, N, 32]` × E `[B, N, 32, 768]` | `[B, N, 768]` |
| **Output** | User representation | - | `[B, N, 768]` |

---

#### Poly Attention 상세

**Step 1: Projection**
```python
h_proj = torch.tanh(self.W_h(history_embeddings))
# Input:  [B, M, 768]
# W_h:    [768, 200]
# Output: [B, M, 200]
```

**Step 2: Attention Logits**
```python
logits = torch.matmul(h_proj, self.context_codes.T) / sqrt(200)
# h_proj:       [B, M, 200]
# context_codes: [32, 200]
# matmul:       [B, M, 200] @ [200, 32] = [B, M, 32]
# scaling:      / 14.14
# Output:       [B, M, 32]
```

**Step 3: Category-Aware Weighting**
```python
# 히스토리 카테고리 임베딩
hist_cat_emb = self.category_embedding(user_category)  # [B, M] → [B, M, 50]
hist_cat_norm = F.normalize(hist_cat_emb, p=2, dim=2)  # [B, M, 50]

# 후보 카테고리 임베딩
cand_cat_emb = self.category_embedding(candidate_category)  # [B, N] → [B, N, 50]
cand_cat_norm = F.normalize(cand_cat_emb, p=2, dim=2)  # [B, N, 50]

# 코사인 유사도
category_sim = torch.bmm(hist_cat_norm, cand_cat_norm.transpose(1, 2))
# [B, M, 50] @ [B, 50, N] = [B, M, N]

# Logits 확장
logits_expanded = logits.unsqueeze(2).expand(-1, -1, N, -1)
# [B, M, 32] → [B, M, 1, 32] → [B, M, N, 32]

# Category bias 추가
category_bias = self.category_aware_lambda * category_sim.unsqueeze(3)
# [B, M, N] → [B, M, N, 1]

logits_final = logits_expanded + category_bias
# [B, M, N, 32]
```

**예시:**
```
히스토리 뉴스 j: 'Lakers win championship' (category: sports)
후보 뉴스 c1: 'Football game today' (category: sports)
후보 뉴스 c2: 'Stock market crash' (category: finance)

cos(sports, sports) = 0.9  →  bias = 0.5 * 0.9 = 0.45
cos(sports, finance) = 0.2  →  bias = 0.5 * 0.2 = 0.1

logits[j, c1, k] += 0.45  (같은 카테고리 → 높은 attention)
logits[j, c2, k] += 0.1   (다른 카테고리 → 낮은 attention)
```

**💡 핵심:**
- **Category-aware**: 후보 뉴스와 카테고리가 유사한 히스토리 뉴스에 더 높은 attention
- **Personalization**: 사용자의 카테고리 선호도 반영
- **GloVe 의존**: `category_embedding`이 의미적으로 초기화되어야 효과적

---

**Step 4: Attention Weights & Weighted Sum**
```python
# Masking
mask_expanded = history_mask.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K)
# [B, M] → [B, M, 1, 1] → [B, M, N, 32]

logits_masked = logits_final.masked_fill(mask_expanded == 0, -1e9)
# Padding 히스토리 제거

# Softmax over history dimension
attn_weights = F.softmax(logits_masked, dim=1)
# [B, M, N, 32] → softmax(dim=1) → [B, M, N, 32]

# Weighted sum
interest_vectors = torch.einsum('bmnk,bmd->bnkd', attn_weights, history_embeddings)
# attn:   [B, M, N, 32]
# hist:   [B, M, 768]
# output: [B, N, 32, 768]
```

**Einsum 상세:**
```
b: batch
m: history news
n: candidate news
k: interest vector index (32)
d: embedding dimension (768)

bmnk, bmd -> bnkd
= for each (b, n, k, d):
    output[b, n, k, d] = Σ_m attn[b, m, n, k] * hist[b, m, d]
```

**💡 핵심:**
- **후보별 관심사**: 각 후보 뉴스에 대해 32개의 interest vectors 생성
- **Soft Selection**: Softmax로 히스토리 뉴스의 가중 평균 계산
- **다양성**: 32개의 context codes가 서로 다른 aspect 캡처

---

#### Disagreement Regularization

```python
def compute_disagreement_loss(self, interest_vectors):
    """
    Args:
        interest_vectors: [B, N, 32, 768]

    Returns:
        loss: scalar
    """
    B, N, K, D = interest_vectors.size()

    # Reshape: [B*N, 32, 768]
    interest_vectors = interest_vectors.view(B*N, K, D)

    # Normalize
    normalized = F.normalize(interest_vectors, p=2, dim=2)  # [B*N, 32, 768]

    # Pairwise cosine similarity
    similarity_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
    # [B*N, 32, 768] @ [B*N, 768, 32] = [B*N, 32, 32]

    # Average over all pairs
    K = interest_vectors.size(1)
    loss = similarity_matrix.sum(dim=(1, 2)) / (K * K)
    # [B*N, 32, 32] → sum → [B*N] → mean → scalar

    return loss.mean()
```

**Similarity Matrix 예시:**
```
[e_1, e_2, ..., e_32]

similarity_matrix[i, j] = cos(e_i, e_j)

이상적인 경우:
  [[1.0, 0.0, 0.0, ..., 0.0],
   [0.0, 1.0, 0.0, ..., 0.0],
   ...
   [0.0, 0.0, 0.0, ..., 1.0]]

평균 = 32 / (32*32) = 0.03125 (최소)

실제 학습 초기:
  평균 ~ 0.2 (높은 유사도)

학습 후:
  평균 ~ 0.05 (낮은 유사도, 다양한 관심사)
```

**Loss 통합:**
```python
total_loss = click_prediction_loss + β * disagreement_loss
           = CrossEntropy(logits, labels) + 0.8 * disagreement_loss
```

**💡 핵심:**
- **다양성 강제**: Interest vectors가 서로 독립적인 관심사를 표현하도록 유도
- **Collapse 방지**: 모든 interest vectors가 동일한 표현으로 수렴하는 것 방지
- **Regularization**: β=0.8로 click prediction과 균형

---

#### Weighted Aggregation

```python
# Step 1: Target-aware transformation
W_e_h_c = F.gelu(self.W_e(candidate_news_representation))
# candidate: [B, N, 768]
# W_e:       [768, 768] + bias
# output:    [B, N, 768]

# Step 2: Attention logits
logits = torch.matmul(
    W_e_h_c.unsqueeze(2),  # [B, N, 1, 768]
    interest_vectors.transpose(2, 3)  # [B, N, 768, 32]
)
# matmul: [B, N, 1, 768] @ [B, N, 768, 32] = [B, N, 1, 32]

logits = logits.squeeze(2)  # [B, N, 32]

# Step 3: Softmax over interest vectors
alpha = F.softmax(logits, dim=2)  # [B, N, 32]

# Step 4: Weighted sum
user_representation = (alpha.unsqueeze(3) * interest_vectors).sum(dim=2)
# alpha:     [B, N, 32, 1]
# interest:  [B, N, 32, 768]
# multiply:  [B, N, 32, 768]
# sum(dim=2): [B, N, 768]
```

**예시:**
```
후보 뉴스 c: 'Lakers win championship'

32개의 interest vectors:
  e_1: 스포츠 관련 (cos(W_e(h_c), e_1) = 0.8)
  e_2: 엔터테인먼트 관련 (cos = 0.3)
  e_3: 정치 관련 (cos = 0.1)
  ...

Softmax 후:
  α_1 = 0.6 (높은 가중치)
  α_2 = 0.2
  α_3 = 0.05
  ...

user_repr = 0.6 * e_1 + 0.2 * e_2 + 0.05 * e_3 + ...
```

**💡 핵심:**
- **Target-aware**: 후보 뉴스와 가장 관련 높은 interest vectors에 집중
- **Dynamic Weighting**: 후보 뉴스마다 다른 관심사 조합 사용
- **Personalization**: 사용자의 다양한 관심사 중 후보와 매칭되는 것 선택

---

### 3.4 전체 파이프라인 예시

#### 입력 데이터

```python
Batch:
  User History (M=3):
    News 1: 'Lakers win NBA championship' (category: sports)
    News 2: 'New iPhone release' (category: tech)
    News 3: 'Movie review: Avengers' (category: entertainment)

  Candidate News (N=2):
    News A: 'Football game highlights' (category: sports)
    News B: 'Stock market update' (category: finance)

Shapes:
  user_title_text: [1, 3, 32]  (PLM token IDs)
  user_category: [1, 3]  (category IDs)
  candidate_news_representation: [1, 2, 768]  (PLM 출력)
  candidate_category: [1, 2]
```

---

#### Forward Pass

**1. History Encoding**
```
PLMMiner(user_title_text):
  Input:  [1, 3, 32]
  BERT:   [3, 32] → [3, 32, 768]
  Pool:   [3, 32, 768] → [3, 768]
  Output: [1, 3, 768]

history_embedding:
  h_1: [768-dim vector for 'Lakers...']
  h_2: [768-dim vector for 'iPhone...']
  h_3: [768-dim vector for 'Avengers...']
```

---

**2. Poly Attention**

```
Projection:
  h_proj_1 = tanh(W_h @ h_1): [200]
  h_proj_2 = tanh(W_h @ h_2): [200]
  h_proj_3 = tanh(W_h @ h_3): [200]

Attention Logits (before category bias):
  logits_1 = h_proj_1 @ c_k^T: [32]  (예: [0.5, 0.3, 0.1, ...])
  logits_2 = h_proj_2 @ c_k^T: [32]
  logits_3 = h_proj_3 @ c_k^T: [32]

Category Similarity:
  cos(sports, sports) = 0.9
  cos(tech, sports) = 0.2
  cos(entertainment, sports) = 0.4
  cos(sports, finance) = 0.1
  cos(tech, finance) = 0.3
  cos(entertainment, finance) = 0.15

Category Bias (λ=0.5):
  For News A (sports):
    bias_1A = 0.5 * 0.9 = 0.45
    bias_2A = 0.5 * 0.2 = 0.1
    bias_3A = 0.5 * 0.4 = 0.2

  For News B (finance):
    bias_1B = 0.5 * 0.1 = 0.05
    bias_2B = 0.5 * 0.3 = 0.15
    bias_3B = 0.5 * 0.15 = 0.075

Final Logits (example for k=0):
  logits[1, A, 0] = 0.5 + 0.45 = 0.95  (h_1에 높은 attention)
  logits[2, A, 0] = 0.3 + 0.1 = 0.4
  logits[3, A, 0] = 0.1 + 0.2 = 0.3

  logits[1, B, 0] = 0.5 + 0.05 = 0.55
  logits[2, B, 0] = 0.3 + 0.15 = 0.45
  logits[3, B, 0] = 0.1 + 0.075 = 0.175

Softmax (over history):
  For News A, k=0:
    α_1 = exp(0.95) / (exp(0.95) + exp(0.4) + exp(0.3)) = 0.6
    α_2 = 0.25
    α_3 = 0.15

  For News B, k=0:
    α_1 = 0.4
    α_2 = 0.35
    α_3 = 0.25

Interest Vectors:
  E_A[0] = 0.6 * h_1 + 0.25 * h_2 + 0.15 * h_3  (스포츠 중심)
  E_B[0] = 0.4 * h_1 + 0.35 * h_2 + 0.25 * h_3  (기술/재무 혼합)
  ... (k=1~31도 동일하게)

Shape:
  interest_vectors: [1, 2, 32, 768]
```

**💡 핵심:**
- News A (sports): h_1 (sports 히스토리)에 높은 attention (0.6)
- News B (finance): h_2 (tech)와 h_1에 비슷한 attention (재무와 관련 없지만 가장 가까운 것 선택)

---

**3. Disagreement Loss**

```
Interest vectors for News A:
  E_A[0], E_A[1], ..., E_A[31]

Pairwise Cosine Similarity:
  cos(E_A[0], E_A[1]) = 0.12
  cos(E_A[0], E_A[2]) = 0.08
  ...

Average: 0.05

Total Loss:
  click_loss + 0.8 * 0.05
```

---

**4. Weighted Aggregation**

```
Target-aware transform:
  W_e_A = gelu(W_e(h_A)): [768]
  W_e_B = gelu(W_e(h_B)): [768]

Attention logits:
  For News A:
    logits_A[k] = W_e_A · E_A[k]
    logits_A = [0.8, 0.3, 0.1, ...]  (k=0이 후보와 가장 관련 높음)

  For News B:
    logits_B = [0.5, 0.6, 0.2, ...]

Softmax:
  α_A = [0.4, 0.2, 0.05, ...]  (k=0에 집중)
  α_B = [0.25, 0.3, 0.1, ...]  (k=1에 집중)

User Representation:
  u_A = 0.4 * E_A[0] + 0.2 * E_A[1] + ...
  u_B = 0.25 * E_B[0] + 0.3 * E_B[1] + ...

Shape:
  user_representation: [1, 2, 768]
```

---

**5. Click Prediction**

```
Dot Product:
  score_A = u_A · h_A = 12.5
  score_B = u_B · h_B = 8.3

Softmax:
  P(click A) = exp(12.5) / (exp(12.5) + exp(8.3)) = 0.98
  P(click B) = 0.02

Prediction: News A (sports)
```

---

### 3.5 주요 하이퍼파라미터 영향

| 파라미터 | 값 | 영향 |
|---------|-----|------|
| **K (num_interest_vectors)** | 32 | 많을수록 다양한 관심사 표현 가능, 계산량 증가 |
| **context_dim** | 200 | Context codes 차원, 클수록 표현력 증가 |
| **category_aware_lambda** | 0.5 | 클수록 카테고리 영향 증가, 0이면 category-agnostic |
| **disagreement_beta** | 0.8 | 클수록 다양성 강제, 너무 크면 click prediction 성능 저하 |
| **plm_frozen_layers** | 10 | 많을수록 PLM 파라미터 고정 (빠르지만 성능↓) |
| **plm_lr** | 1e-5 | PLM fine-tuning learning rate (낮을수록 안정) |

---

### 3.6 GloVe 초기화 vs Random 비교

| 구분 | Random 초기화 | GloVe 초기화 |
|------|--------------|-------------|
| **Category Embedding** | Uniform(-0.1, 0.1) | GloVe 840B 300d (truncate to 50) |
| **코사인 유사도** | 무의미 (random) | 의미적 유사도 반영 |
| **Category-Aware Attention** | 효과 없음 | 의미 있는 가중치 조정 |
| **학습 필요성** | 높음 (scratch) | 낮음 (frozen 가능) |
| **성능** | 낮음 | 높음 (특히 cold-start) |

**예시 비교:**
```
Random:
  cos(sports_emb, entertainment_emb) = -0.03
  cos(sports_emb, finance_emb) = 0.12
  → 무의미한 값

GloVe:
  cos(sports_emb, entertainment_emb) = 0.42
  cos(sports_emb, finance_emb) = 0.15
  → 실제 의미적 유사도 반영
```

---

## 4. 핵심 설계 원칙

### 4.1 PLMMiner 설계

1. **PLM 활용**: BERT로 강력한 텍스트 표현 학습
2. **Minimal Fusion**: Category/SubCategory fusion 없이 순수 PLM 출력만 반환 (MINER가 직접 category embedding 사용)
3. **GloVe Category Embedding**: MINER의 category-aware attention을 위한 의미적 카테고리 표현
4. **Frozen PLM Layers**: 하위 10개층 고정으로 계산 효율 및 overfitting 방지

### 4.2 MINER 설계

1. **Poly Attention**: 32개의 orthogonal context codes로 다양한 관심사 캡처
2. **Category-Aware Attention**: GloVe 초기화된 카테고리 임베딩으로 의미 있는 코사인 유사도 계산
3. **Disagreement Regularization**: Interest vectors 다양성 강제로 관심사 collapse 방지
4. **Weighted Aggregation**: 후보 뉴스별 관련 interest vectors에 동적 가중치 부여

### 4.3 훈련 전략

1. **Differential Learning Rate**: PLM (1e-5) vs 기타 (1e-4)
2. **LR Scheduling**: 10% warmup + linear decay
3. **Auxiliary Loss**: Click prediction + Disagreement regularization
4. **Frozen Category Embedding**: GloVe 의미적 표현 보존

---

## 5. 실험 권장사항

### Ablation Studies

1. **GloVe 효과**:
   ```bash
   # With GloVe
   --use_category_glove

   # Without GloVe (baseline)
   # (플래그 제거)
   ```

2. **Category-Aware Attention 효과**:
   ```bash
   # Full category-aware
   --category_aware_lambda=0.5

   # No category-aware
   --category_aware_lambda=0.0
   ```

3. **Interest Vectors 개수**:
   ```bash
   --num_interest_vectors=16  # vs 32 vs 64
   ```

4. **Aggregation 방식**:
   ```bash
   --miner_aggregation=weighted  # vs max vs mean
   ```

---

**작성 완료**: 2025-12-20
**버전**: 1.0
