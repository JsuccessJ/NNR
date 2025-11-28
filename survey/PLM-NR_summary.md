# PLM-NR 구현 설계서: NNR 코드베이스 통합 방안

## 목차
1. [논문 핵심 분석](#1-논문-핵심-분석)
2. [NNR 코드베이스 구조 분석](#2-nnr-코드베이스-구조-분석)
3. [PLM-NR 통합 전략](#3-plm-nr-통합-전략)
4. [구현 계획: 단계별 상세 설명](#4-구현-계획-단계별-상세-설명)
5. [수정해야 할 파일 목록](#5-수정해야-할-파일-목록)
6. [핵심 기술 이슈](#6-핵심-기술-이슈)
7. [테스트 및 검증 계획](#7-테스트-및-검증-계획)

---

## 1. 논문 핵심 분석

### 1.1 PLM-NR의 핵심 아이디어

**기존 방법의 문제점**:
- NAML, NRMS 등은 **shallow NLP models** 사용 (CNN, Self-Attention)
- GloVe 워드 임베딩에서 학습 시작 → 깊은 의미 이해 부족
- 뉴스 추천 task만의 supervision → 제한적 학습

**PLM-NR의 해결책**:
- **Pre-trained Language Model (PLM)** 활용
  - BERT, RoBERTa, UniLM 등
  - 대규모 unlabeled corpus에서 사전 학습됨
  - 깊은 semantic 정보 포착 가능
- **Fine-tuning** 방식
  - PLM의 마지막 2개 Transformer 레이어만 fine-tune
  - 나머지는 frozen (효율성)
- **Attention Pooling**
  - PLM의 hidden states를 attention으로 통합

### 1.2 아키텍처 구조

```
Input: News Title [w1, w2, ..., wM]
  ↓
PLM (BERT/RoBERTa/UniLM)
  ├─ Token Embedding
  ├─ Transformer Layer 1
  ├─ Transformer Layer 2
  ├─ ...
  └─ Transformer Layer 12
  ↓
Hidden States: [r1, r2, ..., rM]  # [batch*news, seq_len, hidden_dim]
  ↓
Attention Pooling
  ↓
News Embedding: h  # [batch*news, hidden_dim]
```

**수식**:
```
PLM: [r1, r2, ..., rM] = PLM([w1, w2, ..., wM])
Attention: h = Σ(αi * ri), where αi = softmax(v^T * tanh(W*ri + b))
```

### 1.3 실험 결과 핵심

1. **성능 향상**:
   - NAML: 67.78 → UniLM-NAML: 70.50 (AUC)
   - NRMS: 68.18 → UniLM-NRMS: 70.64 (AUC)
   - **약 2.5-3% 절대 성능 향상**

2. **PLM 선택**:
   - UniLM > RoBERTa > BERT
   - 크기: Base (12 layers) 사용 (Medium, Small보다 우수)

3. **Pooling 방법**:
   - Attention > Average > CLS
   - CLS 토큰 단독 사용은 비효율적

4. **Fine-tuning 전략**:
   - 마지막 2개 레이어만 fine-tune
   - 전체 fine-tune과 성능 차이 거의 없음 (효율성 증가)

---

## 2. NNR 코드베이스 구조 분석

### 2.1 현재 News Encoder 구조

**파일**: `newsEncoders.py`

**클래스 계층**:
```python
nn.Module
  ↓
NewsEncoder (base class)
  ├─ word_embedding (GloVe)
  ├─ category_embedding
  ├─ subCategory_embedding
  └─ forward() [abstract method]
  ↓
Concrete Encoders:
  ├─ NAML (Title CNN + Content CNN + Multi-view Attention)
  ├─ CNN (1D Conv + Attention)
  ├─ MHSA (Multi-Head Self-Attention)
  └─ ...
```

**NAML 예시**:
```python
class NAML(NewsEncoder):
    def __init__(self, config):
        super().__init__(config)
        # CNN for title
        self.title_conv = Conv1D(...)
        self.title_attention = Attention(...)
        # CNN for content
        self.content_conv = Conv1D(...)
        self.content_attention = Attention(...)
        # Category encoders
        self.category_affine = nn.Linear(...)
        self.subCategory_affine = nn.Linear(...)
        # Multi-view attention
        self.affine1 = nn.Linear(...)
        self.affine2 = nn.Linear(...)

    def forward(self, title_text, title_mask, ...):
        # 1. Word embedding (GloVe)
        title_w = self.word_embedding(title_text)
        # 2. CNN encoding
        title_c = self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1)
        # 3. Attention pooling
        title_repr = self.title_attention(title_c)
        # 4. Multi-view fusion
        ...
        return news_representation
```

**현재 흐름**:
```
title_text [batch, news, seq_len] (정수 인덱스)
  ↓ word_embedding
[batch, news, seq_len, 300] (GloVe)
  ↓ CNN
[batch, news, seq_len, 400]
  ↓ Attention
[batch, news, 400]
```

### 2.2 데이터 전처리

**파일**: `MIND_corpus.py`

**현재 전처리 흐름**:
1. 정규식 토크나이징: `pat = re.compile(r"[\w]+|[.,!?;|]")`
2. 단어 사전 구축: `word_dict = {'<PAD>': 0, '<UNK>': 1, 'the': 2, ...}`
3. GloVe 임베딩 로딩: `word_embedding_vectors [vocab_size, 300]`
4. 뉴스 텍스트 → 정수 인덱스 변환: `"Lakers win" → [1234, 5678]`

**문제점**:
- PLM은 자체 토크나이저 필요 (WordPiece, BPE)
- PLM은 특수 토큰 필요 (`[CLS]`, `[SEP]`, `[PAD]`)
- 어휘 사전이 다름 (GloVe vs BERT vocab)

---

## 3. PLM-NR 통합 전략

### 3.1 통합 방식 선택

**Option 1: 완전 교체 (논문 방식)**
- 기존 word_embedding 제거
- PLM으로 완전 대체
- 장점: 논문과 동일, 최고 성능
- 단점: 기존 NAML과 호환 불가, 큰 변경

**Option 2: 하이브리드 (추천)**
- 기존 News Encoder 구조 유지
- PLM을 새로운 News Encoder로 추가
- 장점: 기존 코드 보존, 비교 실험 가능
- 단점: 코드 중복 가능성

**선택: Option 2 (하이브리드)**
- `PLMNAML`, `PLMNRMS` 등 새 클래스 생성
- 기존 `NAML`, `NRMS` 유지
- `config.news_encoder`에 `'PLMNAML'` 추가

### 3.2 아키텍처 설계

**새로운 클래스 계층**:
```python
nn.Module
  ↓
PLMNewsEncoder (새 base class)
  ├─ plm (BERT/RoBERTa/UniLM)
  ├─ attention_pooling
  ├─ category_embedding (선택적)
  └─ forward()
  ↓
Concrete PLM Encoders:
  ├─ PLMNAML (PLM + Category Fusion)
  ├─ PLMNRMS (PLM only)
  ├─ PLMNPA (PLM + Personalized Attention)
  └─ ...
```

### 3.3 핵심 변경 사항

**1. 데이터 전처리 레이어 추가**:
```python
class PLMTokenizer:
    """PLM용 토크나이저 래퍼"""
    def __init__(self, plm_name):
        if plm_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif plm_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        ...

    def encode(self, text, max_length):
        """텍스트 → 토큰 ID + attention mask"""
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']
```

**2. News Encoder 구현**:
```python
class PLMNAML(nn.Module):
    def __init__(self, config):
        super().__init__()
        # PLM 로딩
        if config.plm_type == 'bert':
            self.plm = BertModel.from_pretrained('bert-base-uncased')
        elif config.plm_type == 'roberta':
            self.plm = RobertaModel.from_pretrained('roberta-base')

        # PLM frozen (마지막 2개 레이어만 학습)
        for name, param in self.plm.named_parameters():
            if 'encoder.layer.10' not in name and 'encoder.layer.11' not in name:
                param.requires_grad = False

        # Attention Pooling
        self.attention = Attention(hidden_dim=768, attention_dim=200)

        # Category Fusion (NAML 스타일)
        self.category_embedding = nn.Embedding(config.category_num, 50)
        self.subCategory_embedding = nn.Embedding(config.subCategory_num, 50)

        self.news_embedding_dim = 768 + 50 + 50  # PLM + category + subCategory

    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num

        # 1. PLM encoding
        title_text = title_text.view([batch_news_num, -1])  # [B*N, seq_len]
        title_mask = title_mask.view([batch_news_num, -1])  # [B*N, seq_len]

        plm_output = self.plm(input_ids=title_text, attention_mask=title_mask)
        hidden_states = plm_output.last_hidden_state  # [B*N, seq_len, 768]

        # 2. Attention Pooling
        news_repr = self.attention(hidden_states, mask=title_mask)  # [B*N, 768]
        news_repr = news_repr.view([batch_size, news_num, 768])

        # 3. Category Fusion
        category_repr = self.category_embedding(category)  # [B, N, 50]
        subCategory_repr = self.subCategory_embedding(subCategory)  # [B, N, 50]

        news_representation = torch.cat([news_repr, category_repr, subCategory_repr], dim=2)  # [B, N, 868]

        return news_representation
```

---

## 4. 구현 계획: 단계별 상세 설명

### Phase 1: 환경 설정 및 의존성 추가

**목표**: PLM 라이브러리 설치 및 설정

**작업 내용**:

1. **requirements.txt 업데이트**:
```txt
torch>=1.11.0
transformers>=4.20.0  # HuggingFace Transformers
tokenizers>=0.12.0
sentencepiece>=0.1.96  # 일부 PLM에서 필요
```

2. **config.py에 PLM 관련 설정 추가**:
```python
class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser()
        # 기존 인자들...

        # PLM 관련 인자 추가
        parser.add_argument('--plm_type', type=str, default='bert',
                          choices=['bert', 'roberta', 'unilm', 'none'],
                          help='Pre-trained Language Model type')
        parser.add_argument('--plm_model_name', type=str, default='bert-base-uncased',
                          help='Specific PLM model name from HuggingFace')
        parser.add_argument('--plm_frozen_layers', type=int, default=10,
                          help='Number of PLM layers to freeze (0=fine-tune all)')
        parser.add_argument('--plm_lr', type=float, default=1e-5,
                          help='Learning rate for PLM fine-tuning')
        parser.add_argument('--plm_pooling', type=str, default='attention',
                          choices=['cls', 'average', 'attention'],
                          help='Pooling method for PLM hidden states')
        parser.add_argument('--use_plm_news_encoder', action='store_true',
                          help='Use PLM-based news encoder')
```

3. **디렉토리 구조 생성**:
```bash
NNR/
├── plm_cache/          # PLM 모델 캐시 디렉토리
├── plm_tokenizers/     # 토크나이저 캐시
└── plm_models/         # 체크포인트 저장
```

### Phase 2: PLM 토크나이저 통합

**목표**: 기존 전처리 파이프라인에 PLM 토크나이저 추가

**파일**: `MIND_corpus.py`

**상세 구현**:

```python
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

class MIND_Corpus:
    def __init__(self, config: Config):
        # 기존 전처리 (GloVe 기반)
        MIND_Corpus.preprocess(config)

        # 기존 데이터 로딩
        with open('user_ID-%s.json' % config.dataset, 'r') as f:
            self.user_ID_dict = json.load(f)
        # ... (기존 코드)

        # PLM 사용 시 추가 전처리
        if config.use_plm_news_encoder:
            self._preprocess_for_plm(config)

    def _preprocess_for_plm(self, config: Config):
        """PLM용 토큰화 및 저장"""
        print('Preprocessing news texts for PLM...')

        # 1. PLM 토크나이저 로딩
        if config.plm_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')
        elif config.plm_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.plm_model_name, cache_dir='plm_cache/')

        # 2. 뉴스 제목 토큰화
        plm_title_file = f'plm_title_{config.plm_type}_{config.dataset}.pkl'

        if not os.path.exists(plm_title_file):
            news_plm_title_ids = []
            news_plm_title_masks = []

            for news_index in tqdm(range(self.news_num)):
                # 원본 텍스트 가져오기 (news_raw.tsv에서)
                # 실제로는 self.news_title_text를 역변환하거나 원본 유지 필요
                title_text = self._get_news_title_text(news_index)

                # PLM 토크나이징
                encoding = tokenizer(
                    title_text,
                    max_length=config.max_title_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='np'
                )

                news_plm_title_ids.append(encoding['input_ids'][0])
                news_plm_title_masks.append(encoding['attention_mask'][0])

            # NumPy 배열로 변환
            self.news_plm_title_ids = np.array(news_plm_title_ids, dtype=np.int32)
            self.news_plm_title_masks = np.array(news_plm_title_masks, dtype=np.int32)

            # 저장
            with open(plm_title_file, 'wb') as f:
                pickle.dump({
                    'title_ids': self.news_plm_title_ids,
                    'title_masks': self.news_plm_title_masks
                }, f)
        else:
            # 로딩
            with open(plm_title_file, 'rb') as f:
                data = pickle.load(f)
                self.news_plm_title_ids = data['title_ids']
                self.news_plm_title_masks = data['title_masks']

        print(f'PLM tokenization completed. Shape: {self.news_plm_title_ids.shape}')

    def _get_news_title_text(self, news_index):
        """뉴스 인덱스로부터 원본 텍스트 추출"""
        # Option 1: 원본 텍스트 저장했다면
        if hasattr(self, 'news_title_texts'):
            return self.news_title_texts[news_index]

        # Option 2: word_dict 역변환 (정확도 떨어질 수 있음)
        word_indices = self.news_title_text[news_index]
        reverse_word_dict = {v: k for k, v in self.word_dict.items()}
        words = [reverse_word_dict.get(idx, '<UNK>') for idx in word_indices if idx > 0]
        return ' '.join(words)
```

**핵심 이슈 1: 원본 텍스트 보존**

현재 `MIND_corpus.py`는 원본 텍스트를 정수 인덱스로만 저장합니다. PLM은 원본 텍스트가 필요합니다.

**해결 방안**:
```python
# MIND_corpus.py의 news meta data 생성 부분 수정
class MIND_Corpus:
    def __init__(self, config: Config):
        # ... 기존 코드 ...

        # 원본 텍스트 저장 (PLM용)
        self.news_title_texts = [''] * self.news_num  # 추가

        for line in news_lines:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
            index = self.news_ID_dict[news_ID]

            # 원본 텍스트 저장 (추가)
            self.news_title_texts[index] = title

            # 기존 정수 인덱스 변환 (유지)
            words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                # ... (기존 코드)
```

### Phase 3: PLM News Encoder 구현

**파일**: `plmNewsEncoders.py` (새 파일)

**상세 구현**:

```python
"""
PLM-based News Encoders for News Recommendation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, AutoModel
from config import Config
from layers import Attention


class PLMNewsEncoder(nn.Module):
    """Base class for PLM-based news encoders"""

    def __init__(self, config: Config):
        super(PLMNewsEncoder, self).__init__()
        self.config = config

        # 1. PLM 로딩
        self.plm_hidden_dim = 768  # BERT-Base hidden size
        if config.plm_type == 'bert':
            self.plm = BertModel.from_pretrained(
                config.plm_model_name,
                cache_dir='plm_cache/'
            )
            self.plm_hidden_dim = self.plm.config.hidden_size
        elif config.plm_type == 'roberta':
            self.plm = RobertaModel.from_pretrained(
                config.plm_model_name,
                cache_dir='plm_cache/'
            )
            self.plm_hidden_dim = self.plm.config.hidden_size
        else:
            self.plm = AutoModel.from_pretrained(
                config.plm_model_name,
                cache_dir='plm_cache/'
            )
            self.plm_hidden_dim = self.plm.config.hidden_size

        # 2. PLM Layer Freezing (마지막 N개 레이어만 학습)
        self._freeze_plm_layers(config.plm_frozen_layers)

        # 3. Pooling 방법 선택
        self.pooling_method = config.plm_pooling
        if self.pooling_method == 'attention':
            self.attention = Attention(
                hidden_dim=self.plm_hidden_dim,
                attention_dim=config.attention_dim
            )

        # 4. Dropout
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=False)

        # 5. Auxiliary loss (일부 모델에서 사용)
        self.auxiliary_loss = None

    def _freeze_plm_layers(self, frozen_layers):
        """PLM의 하위 레이어 frozen"""
        if frozen_layers == 0:
            # 전체 fine-tune
            return

        # BERT/RoBERTa 구조: encoder.layer.0 ~ encoder.layer.11
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
            # [CLS] 토큰의 representation 사용
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

        # 4. Category fusion
        category_repr = self.dropout(self.category_embedding(category))  # [B, N, cat_dim]
        subCategory_repr = self.dropout(self.subCategory_embedding(subCategory))  # [B, N, subcat_dim]

        # 5. Concatenation
        news_representation = torch.cat([
            news_repr,
            category_repr,
            subCategory_repr
        ], dim=2)  # [B, N, 768+50+50=868]

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
```

### Phase 4: Dataset 수정

**파일**: `MIND_dataset.py`

**목표**: PLM 토큰 ID와 attention mask 반환

**수정 사항**:

```python
class MIND_Train_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus):
        # 기존 초기화
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        # ... (기존 필드들)

        # PLM 데이터 추가
        self.use_plm = hasattr(corpus, 'news_plm_title_ids')
        if self.use_plm:
            self.news_plm_title_ids = corpus.news_plm_title_ids
            self.news_plm_title_masks = corpus.news_plm_title_masks

    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]
        sample_index = self.train_samples[index]
        behavior_index = train_behavior[5]

        if self.use_plm:
            # PLM 데이터 반환
            return (
                train_behavior[0],  # user_ID
                self.news_category[history_index],  # user history category
                self.news_subCategory[history_index],
                self.news_plm_title_ids[history_index],  # PLM token IDs (history)
                self.news_plm_title_masks[history_index],  # PLM attention masks (history)
                self.news_title_entity[history_index],  # entity는 유지 (사용 안 할 수도 있음)
                # content는 생략 (title만 사용)
                self.news_abstract_text[history_index],  # placeholder
                self.news_abstract_mask[history_index],
                self.news_abstract_entity[history_index],
                train_behavior[2],  # user_history_mask
                self.user_history_graph[behavior_index],
                self.user_history_category_mask[behavior_index],
                self.user_history_category_indices[behavior_index],
                # Candidate news
                self.news_category[sample_index],
                self.news_subCategory[sample_index],
                self.news_plm_title_ids[sample_index],  # PLM token IDs (candidates)
                self.news_plm_title_masks[sample_index],  # PLM attention masks
                self.news_title_entity[sample_index],
                self.news_abstract_text[sample_index],
                self.news_abstract_mask[sample_index],
                self.news_abstract_entity[sample_index]
            )
        else:
            # 기존 데이터 반환 (변경 없음)
            return (
                train_behavior[0],
                self.news_category[history_index],
                # ... (기존 코드)
            )
```

### Phase 5: Model 클래스 수정

**파일**: `model.py`

**수정 사항**:

```python
from config import Config
import torch
import torch.nn as nn
import newsEncoders
import userEncoders
import plmNewsEncoders  # 추가

class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()

        # News Encoder 선택
        if config.use_plm_news_encoder:
            # PLM 기반 News Encoder
            if config.news_encoder == 'NAML':
                self.news_encoder = plmNewsEncoders.PLMNAML(config)
            elif config.news_encoder == 'NRMS':
                self.news_encoder = plmNewsEncoders.PLMNRMS(config)
            elif config.news_encoder == 'NPA':
                self.news_encoder = plmNewsEncoders.PLMNPA(config)
            else:
                raise Exception(f'PLM-{config.news_encoder} is not implemented')
        else:
            # 기존 News Encoder (GloVe 기반)
            if config.news_encoder == 'CNE':
                self.news_encoder = newsEncoders.CNE(config)
            elif config.news_encoder == 'CNN':
                self.news_encoder = newsEncoders.CNN(config)
            # ... (기존 코드)

        # User Encoder (변경 없음)
        if config.user_encoder == 'SUE':
            self.user_encoder = userEncoders.SUE(self.news_encoder, config)
        # ... (기존 코드)

        # 나머지 초기화 (변경 없음)
        self.model_name = config.news_encoder + '-' + config.user_encoder
        # ...
```

### Phase 6: Trainer 수정

**파일**: `trainer.py`

**목표**: PLM용 learning rate 분리, gradient accumulation 추가

**수정 사항**:

```python
class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
        self.model = model
        # ... (기존 초기화)

        # Optimizer 수정: PLM과 나머지 파라미터 learning rate 분리
        if config.use_plm_news_encoder:
            # PLM 파라미터와 나머지 파라미터 분리
            plm_params = []
            other_params = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'plm' in name:
                        plm_params.append(param)
                    else:
                        other_params.append(param)

            # 서로 다른 learning rate
            self.optimizer = optim.Adam([
                {'params': plm_params, 'lr': config.plm_lr},  # 1e-5
                {'params': other_params, 'lr': config.lr}     # 1e-4
            ], weight_decay=config.weight_decay)

            print(f'PLM parameters: {len(plm_params)}, Other parameters: {len(other_params)}')
        else:
            # 기존 방식
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.lr,
                weight_decay=config.weight_decay
            )

        # Gradient Accumulation (옵션)
        self.accumulation_steps = config.accumulation_steps if hasattr(config, 'accumulation_steps') else 1
```

**훈련 루프 수정**:

```python
def train(self):
    model = self.model
    for e in tqdm(range(1, self.epoch + 1)):
        self.train_dataset.negative_sampling()
        train_dataloader = DataLoader(...)
        model.train()
        epoch_loss = 0

        # Gradient accumulation 추가
        self.optimizer.zero_grad()

        for batch_idx, (user_ID, user_category, ...) in enumerate(train_dataloader):
            # GPU 전송
            user_ID = user_ID.cuda(non_blocking=True)
            # ...

            # Forward
            logits = model(...)

            # Loss 계산
            loss = self.loss(logits)
            # Auxiliary loss
            if model.news_encoder.auxiliary_loss is not None:
                loss += model.news_encoder.auxiliary_loss.mean()

            # Gradient accumulation을 위한 loss scaling
            loss = loss / self.accumulation_steps
            epoch_loss += float(loss) * user_ID.size(0) * self.accumulation_steps

            # Backward
            loss.backward()

            # Gradient accumulation step마다 update
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # 마지막 배치 처리
        if (batch_idx + 1) % self.accumulation_steps != 0:
            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # 나머지 코드 (validation 등) - 변경 없음
        # ...
```

---

## 5. 수정해야 할 파일 목록

### 5.1 새로 생성할 파일

1. **`plmNewsEncoders.py`** (핵심)
   - `PLMNewsEncoder` (base class)
   - `PLMNAML`
   - `PLMNRMS`
   - `PLMNPA`
   - `PLMLSTUR`

### 5.2 수정할 파일

1. **`config.py`**
   - PLM 관련 인자 추가
   - `plm_type`, `plm_model_name`, `plm_frozen_layers`, `plm_lr`, `plm_pooling`, `use_plm_news_encoder`

2. **`MIND_corpus.py`**
   - `_preprocess_for_plm()` 메서드 추가
   - 원본 텍스트 저장 (`news_title_texts`)
   - PLM 토큰화 및 캐싱

3. **`MIND_dataset.py`**
   - `__init__()`: PLM 데이터 로딩
   - `__getitem__()`: PLM token IDs와 masks 반환

4. **`model.py`**
   - News Encoder 선택 로직에 PLM 분기 추가

5. **`trainer.py`**
   - Optimizer 수정 (learning rate 분리)
   - Gradient accumulation 추가 (옵션)

6. **`requirements.txt`**
   - `transformers>=4.20.0` 추가

### 5.3 수정하지 않을 파일

1. **`userEncoders.py`**: 변경 불필요 (News Encoder와 독립적)
2. **`layers.py`**: 변경 불필요 (Attention 등 재사용)
3. **`util.py`**: 변경 불필요 (평가 로직 동일)
4. **`evaluate.py`**: 변경 불필요

---

## 6. 핵심 기술 이슈

### 6.1 메모리 관리

**문제**: PLM은 매우 큼 (BERT-Base: 109M 파라미터)

**해결책**:

1. **Gradient Checkpointing**:
```python
# plmNewsEncoders.py
class PLMNewsEncoder(nn.Module):
    def __init__(self, config):
        # ...
        if config.use_gradient_checkpointing:
            self.plm.gradient_checkpointing_enable()
```
- 메모리 사용량 감소 (약 40%)
- 속도는 약간 느려짐 (약 20%)

2. **Mixed Precision Training**:
```python
# trainer.py
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, ...):
        # ...
        self.scaler = GradScaler() if config.use_amp else None

    def train(self):
        for batch in train_dataloader:
            with autocast():
                logits = model(...)
                loss = self.loss(logits)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
```

3. **배치 크기 조정**:
- 기존: `batch_size=64`
- PLM: `batch_size=16` 또는 `32`
- Gradient accumulation으로 effective batch size 유지

### 6.2 토크나이저 불일치

**문제**: GloVe와 PLM의 어휘가 다름

**현재 데이터 흐름**:
```
원본 텍스트: "Lakers win championship"
  ↓ 정규식 토크나이징
['lakers', 'win', 'championship']
  ↓ word_dict
[1234, 5678, 9012]
  ↓ 저장 (self.news_title_text)
```

**PLM 데이터 흐름**:
```
원본 텍스트: "Lakers win championship"
  ↓ BERT Tokenizer (WordPiece)
['[CLS]', 'lakers', 'win', 'championship', '[SEP]']
  ↓ BERT vocab
[101, 19837, 2663, 2528, 102]
  ↓ 저장 (self.news_plm_title_ids)
```

**해결책**:
- 원본 텍스트 보존 필수
- `MIND_corpus.py`에서 `news_title_texts` 추가
- 전처리 시 원본 저장:
```python
self.news_title_texts = [''] * self.news_num

for line in news_lines:
    news_ID, category, subCategory, title, abstract, ... = line.split('\t')
    index = self.news_ID_dict[news_ID]
    self.news_title_texts[index] = title  # 원본 저장
```

### 6.3 특수 토큰 처리

**PLM 특수 토큰**:
- `[CLS]`: 문장 시작 (classification token)
- `[SEP]`: 문장 구분
- `[PAD]`: 패딩
- `[MASK]`: 마스킹 (우리는 사용 안 함)

**주의사항**:
```python
# 올바른 예
tokenizer("Lakers win", max_length=10, padding='max_length')
# Output: [CLS] lakers win [SEP] [PAD] [PAD] ...

# 잘못된 예: 수동으로 [CLS] 추가하지 말 것
text = "[CLS] Lakers win [SEP]"  # X
tokenizer(text)  # [[CLS], [, CLS, ], lakers, ...]  잘못됨
```

### 6.4 Attention Mask

**역할**: 패딩 토큰 무시

**중요성**:
```python
# Attention mask가 없으면
hidden_states = [h1, h2, h3, h_pad, h_pad]  # 모두 동일하게 처리
# 평균: (h1+h2+h3+h_pad+h_pad)/5  # 패딩 포함 (잘못됨)

# Attention mask 사용
mask = [1, 1, 1, 0, 0]
masked_hidden = hidden_states * mask
# 평균: (h1+h2+h3)/3  # 패딩 제외 (올바름)
```

**구현**:
```python
def _pool_hidden_states(self, hidden_states, attention_mask):
    if self.pooling_method == 'average':
        # Mask 확장: [B, seq] → [B, seq, hidden]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Masking 적용
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # 0 방지

        news_embedding = sum_hidden / sum_mask
        return news_embedding
```

### 6.5 Fine-tuning 전략

**논문 권장사항**:
- 마지막 2개 레이어만 fine-tune
- 나머지는 frozen

**구현**:
```python
def _freeze_plm_layers(self, frozen_layers):
    """
    BERT-Base: 12 layers (0~11)
    frozen_layers=10 → layer 0~9 frozen, layer 10~11 학습
    """
    total_layers = len(self.plm.encoder.layer)

    for layer_idx in range(frozen_layers):
        for param in self.plm.encoder.layer[layer_idx].parameters():
            param.requires_grad = False

    # Embedding layer도 frozen
    for param in self.plm.embeddings.parameters():
        param.requires_grad = False
```

**Learning Rate**:
- PLM layers: `1e-5` (매우 작음, catastrophic forgetting 방지)
- 다른 layers: `1e-4` (기존 learning rate)

### 6.6 Multilingual 지원

**논문**: InfoXLM, Unicoder 사용

**구현**:
```python
# config.py
parser.add_argument('--plm_type', choices=[
    'bert',           # 영어
    'roberta',        # 영어
    'xlm-roberta',    # 다국어
    'infoxlm'         # 다국어 (더 좋음)
])

# plmNewsEncoders.py
if config.plm_type == 'xlm-roberta':
    self.plm = AutoModel.from_pretrained('xlm-roberta-base')
elif config.plm_type == 'infoxlm':
    # InfoXLM은 HuggingFace에 공식 모델이 없을 수 있음
    # 별도 체크포인트 다운로드 필요
    self.plm = AutoModel.from_pretrained('microsoft/infoxlm-base')
```

---

## 7. 테스트 및 검증 계획

### 7.1 Unit Test

**Test 1: PLM 토크나이저**
```python
def test_plm_tokenizer():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text = "Lakers win championship"
    encoding = tokenizer(text, max_length=10, padding='max_length', truncation=True, return_tensors='pt')

    print("Input IDs:", encoding['input_ids'])
    print("Attention Mask:", encoding['attention_mask'])

    # 예상 출력:
    # Input IDs: [[101, 19837, 2663, 2528, 102, 0, 0, 0, 0, 0]]
    # Attention Mask: [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
```

**Test 2: PLM Forward Pass**
```python
def test_plm_forward():
    config = Config()
    config.plm_type = 'bert'
    config.plm_model_name = 'bert-base-uncased'
    config.use_plm_news_encoder = True

    model = plmNewsEncoders.PLMNAML(config)

    # 더미 입력
    batch_size, news_num, seq_len = 2, 3, 10
    title_text = torch.randint(0, 1000, (batch_size, news_num, seq_len))
    title_mask = torch.ones(batch_size, news_num, seq_len)
    category = torch.randint(0, 10, (batch_size, news_num))
    subCategory = torch.randint(0, 20, (batch_size, news_num))

    # Forward
    output = model(title_text, title_mask, None, None, None, None, category, subCategory, None)

    print("Output shape:", output.shape)
    # 예상: [2, 3, 868] (768 + 50 + 50)
    assert output.shape == (batch_size, news_num, model.news_embedding_dim)
```

**Test 3: Frozen Layers 확인**
```python
def test_frozen_layers():
    config = Config()
    config.plm_frozen_layers = 10
    model = plmNewsEncoders.PLMNAML(config)

    # Layer 0~9: frozen
    for layer_idx in range(10):
        for param in model.plm.encoder.layer[layer_idx].parameters():
            assert param.requires_grad == False, f"Layer {layer_idx} should be frozen"

    # Layer 10~11: trainable
    for layer_idx in range(10, 12):
        for param in model.plm.encoder.layer[layer_idx].parameters():
            assert param.requires_grad == True, f"Layer {layer_idx} should be trainable"

    print("Frozen layers test passed!")
```

### 7.2 Integration Test

**Test 4: End-to-End 훈련**
```bash
# 작은 데이터셋으로 빠른 테스트
python main.py \
  --dataset small \
  --news_encoder NAML \
  --user_encoder ATT \
  --use_plm_news_encoder \
  --plm_type bert \
  --plm_model_name bert-base-uncased \
  --plm_frozen_layers 10 \
  --epoch 1 \
  --batch_size 16
```

**Test 5: 기존 모델과 비교**
```bash
# 기존 NAML
python main.py --news_encoder NAML --user_encoder ATT --epoch 5

# PLM-NAML
python main.py --news_encoder NAML --user_encoder ATT --use_plm_news_encoder --plm_type bert --epoch 5

# 성능 비교 (AUC, MRR, nDCG)
```

### 7.3 성능 검증

**예상 결과** (MIND 데이터셋):
```
기존 NAML:           AUC=67.78, MRR=33.24
BERT-NAML (예상):    AUC=69.50, MRR=34.70 (+1.7%)
RoBERTa-NAML (예상): AUC=69.60, MRR=34.80 (+1.8%)
UniLM-NAML (예상):   AUC=70.50, MRR=35.30 (+2.7%)
```

**검증 체크리스트**:
- [ ] AUC 향상 확인 (최소 +1.5%)
- [ ] MRR, nDCG@5, nDCG@10 모두 향상
- [ ] 훈련 시간 증가폭 확인 (2~3배 예상)
- [ ] 메모리 사용량 확인 (GPU 16GB로 가능한지)
- [ ] Overfitting 확인 (validation loss 모니터링)

---

## 요약: 구현 체크리스트

### Phase 1: 환경 설정 ✓
- [ ] `transformers` 라이브러리 설치
- [ ] `config.py`에 PLM 인자 추가
- [ ] PLM 캐시 디렉토리 생성

### Phase 2: 데이터 전처리 ✓
- [ ] `MIND_corpus.py`: 원본 텍스트 저장 로직 추가
- [ ] `MIND_corpus.py`: `_preprocess_for_plm()` 구현
- [ ] PLM 토큰화 결과 캐싱

### Phase 3: PLM Encoder 구현 ✓
- [ ] `plmNewsEncoders.py` 파일 생성
- [ ] `PLMNewsEncoder` base class 구현
- [ ] `PLMNAML` 구현
- [ ] `PLMNRMS` 구현

### Phase 4: Dataset 수정 ✓
- [ ] `MIND_dataset.py`: PLM 데이터 로딩
- [ ] `__getitem__()`: PLM token IDs 반환

### Phase 5: Model 통합 ✓
- [ ] `model.py`: PLM Encoder 선택 로직

### Phase 6: Trainer 수정 ✓
- [ ] `trainer.py`: Learning rate 분리
- [ ] Gradient accumulation 추가 (옵션)

### Phase 7: 테스트 ✓
- [ ] Unit tests 작성 및 실행
- [ ] End-to-end 훈련 테스트
- [ ] 성능 비교 실험

---

## 최종 노트

**핵심 포인트**:
1. **기존 코드 보존**: 하이브리드 방식으로 기존 NAML, NRMS와 공존
2. **원본 텍스트 필수**: PLM은 원본 텍스트 필요 → `MIND_corpus.py` 수정 필수
3. **Learning Rate 조정**: PLM은 1e-5, 다른 layers는 1e-4
4. **메모리 최적화**: Gradient checkpointing, Mixed precision, 작은 배치 크기
5. **Fine-tuning 전략**: 마지막 2개 레이어만 학습 (frozen_layers=10)

**예상 성능**:
- 논문 기준: NAML 67.78% → UniLM-NAML 70.50% (AUC)
- 약 2.7% 절대 성능 향상
- 훈련 시간: 2~3배 증가
- GPU 메모리: 기존 8GB → PLM 16GB 필요

**추가 고려사항**:
- Online serving: PLM은 너무 무거움 → Knowledge Distillation 고려
- Inference 최적화: ONNX, TensorRT 변환
- 모델 압축: Pruning, Quantization
