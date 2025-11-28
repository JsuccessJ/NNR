# NAML 모델 훈련/평가/저장 완벽 가이드

## 목차
1. [시스템 개요](#1-시스템-개요)
2. [데이터 전처리 파이프라인](#2-데이터-전처리-파이프라인)
3. [데이터셋 클래스와 데이터 로더](#3-데이터셋-클래스와-데이터-로더)
4. [NAML 모델 아키텍처](#4-naml-모델-아키텍처)
5. [훈련 과정 (Forward & Backward)](#5-훈련-과정-forward--backward)
6. [평가 과정](#6-평가-과정)
7. [모델 저장 및 체크포인트 관리](#7-모델-저장-및-체크포인트-관리)
8. [전체 실행 흐름](#8-전체-실행-흐름)

---

## 1. 시스템 개요

### 1.1 프로젝트 구조
```
NNR/
├── config.py              # 설정 관리
├── MIND_corpus.py         # 데이터 전처리 및 코퍼스 관리
├── MIND_dataset.py        # PyTorch Dataset 클래스
├── model.py               # 메인 모델 래퍼
├── newsEncoders.py        # 뉴스 인코더들 (NAML 포함)
├── userEncoders.py        # 유저 인코더들
├── trainer.py             # 훈련 로직
├── util.py                # 평가 및 유틸리티 함수
├── main.py                # 메인 실행 파일
└── layers.py              # 커스텀 레이어들
```

### 1.2 NAML이란?
**NAML (Neural news recommendation with Attentive Multi-view Learning)**은 뉴스 추천 모델로, 뉴스의 여러 측면(제목, 내용, 카테고리, 서브카테고리)을 개별적으로 인코딩한 후 어텐션 메커니즘으로 통합합니다.

### 1.3 핵심 개념
- **News Encoder**: 뉴스 기사를 고정 길이 벡터로 인코딩
- **User Encoder**: 사용자의 뉴스 읽기 이력을 사용자 표현 벡터로 인코딩
- **Click Predictor**: 사용자-뉴스 쌍의 클릭 확률 예측

---

## 2. 데이터 전처리 파이프라인

### 2.1 파일 위치: `MIND_corpus.py`

### 2.2 전처리 흐름도
```
원본 데이터 (behaviors_raw.tsv, news_raw.tsv)
    ↓
사전 구축 (user_ID, news_ID, category, vocabulary, entity)
    ↓
워드 임베딩 로딩 (GloVe)
    ↓
엔티티 임베딩 로딩
    ↓
유저 히스토리 그래프 생성
    ↓
MIND_Corpus 객체 생성
```

### 2.3 상세 코드 분석

#### 2.3.1 클래스 정의 및 초기화
**파일**: `MIND_corpus.py:23-223`

```python
class MIND_Corpus:
    @staticmethod
    def preprocess(config: Config):
```

**클래스 설명**:
- `MIND_Corpus`: MIND 데이터셋의 모든 전처리를 담당하는 클래스
- `@staticmethod`: 인스턴스 생성 없이 호출 가능한 정적 메서드
- 파이썬의 `@staticmethod` 데코레이터는 클래스 메서드를 정적 메서드로 만듦

**핵심 파이썬 개념**:
- `staticmethod`: self 파라미터 없이 호출 가능, 클래스나 인스턴스 상태에 접근 불가
- `Config` 타입 힌트: config 파라미터는 Config 클래스의 인스턴스여야 함

#### 2.3.2 사용자 ID 사전 구축
**파일**: `MIND_corpus.py:48-55`

```python
# 1. user ID dictionary
with open(os.path.join(config.train_root, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
    for line in train_behaviors_f:
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        if user_ID not in user_ID_dict:
            user_ID_dict[user_ID] = len(user_ID_dict)
    with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
        json.dump(user_ID_dict, user_ID_f)
```

**단계별 분석**:

1. **파일 열기**:
   - `with open(...)`: 컨텍스트 매니저, 파일 자동 닫기 보장
   - `os.path.join()`: OS에 독립적인 경로 결합
   - `'r'`: 읽기 모드
   - `encoding='utf-8'`: UTF-8 인코딩으로 읽기

2. **라인 파싱**:
   ```python
   impression_ID, user_ID, time, history, impressions = line.split('\t')
   ```
   - **입력**: `"123\tU12345\t2019-11-13 10:00:00\tN1 N2 N3\tN4-1 N5-0"`
   - **출력**:
     - `impression_ID = "123"`
     - `user_ID = "U12345"`
     - `time = "2019-11-13 10:00:00"`
     - `history = "N1 N2 N3"` (과거 읽은 뉴스)
     - `impressions = "N4-1 N5-0"` (추천된 뉴스-클릭여부)

3. **사전 구축**:
   ```python
   if user_ID not in user_ID_dict:
       user_ID_dict[user_ID] = len(user_ID_dict)
   ```
   - **목적**: 각 사용자에게 고유한 정수 ID 할당
   - **예시**:
     - 첫 번째 사용자 "U12345" → 0
     - 두 번째 사용자 "U67890" → 1
   - **len(user_ID_dict)**: 현재까지 추가된 사용자 수 = 다음 할당할 ID

4. **JSON 저장**:
   ```python
   json.dump(user_ID_dict, user_ID_f)
   ```
   - Python 딕셔너리를 JSON 파일로 직렬화
   - **출력 예시**: `{"<UNK>": 0, "U12345": 1, "U67890": 2}`

#### 2.3.3 뉴스 ID 및 텍스트 전처리
**파일**: `MIND_corpus.py:58-102`

```python
for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
    with open(os.path.join(prefix, 'news_raw.tsv'), 'r', encoding='utf-8') as news_f:
        for line in news_f:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
```

**단계별 분석**:

1. **데이터 분할 순회**:
   ```python
   for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
   ```
   - `enumerate()`: 인덱스와 값을 함께 반환
   - **i=0**: train, **i=1**: dev, **i=2**: test
   - 이 순서가 중요한 이유: 훈련 데이터에서만 새로운 단어 추가

2. **뉴스 라인 파싱**:
   - **입력 예시**: `"N54321\tSports\tFootball\tLakers win championship\tThe Lakers won...\t\t[{...}]\t[{...}]"`
   - **출력**:
     - `news_ID = "N54321"`
     - `category = "Sports"`
     - `subCategory = "Football"`
     - `title = "Lakers win championship"`
     - `abstract = "The Lakers won..."`

3. **토크나이징**:
   ```python
   words = pat.findall(title.lower()) if config.tokenizer == 'MIND' else word_tokenize(title.lower())
   ```
   - **정규식 패턴**: `pat = re.compile(r"[\w]+|[.,!?;|]")`
     - `[\w]+`: 하나 이상의 단어 문자 (알파벳, 숫자, _)
     - `|`: 또는
     - `[.,!?;|]`: 구두점
   - **입력**: `"Lakers win championship!"`
   - **출력**: `["lakers", "win", "championship", "!"]`

4. **단어 빈도수 집계**:
   ```python
   for word in words:
       if is_number(word):
           word_counter['<NUM>'] += 1
       else:
           if i == 0:  # training set
               word_counter[word] += 1
           else:
               if word in word_counter:
                   word_counter[word] += 1
   ```

   **로직 설명**:
   - **숫자 처리**: 모든 숫자를 `<NUM>` 토큰으로 통일
   - **훈련 데이터 (i==0)**: 모든 단어 추가
   - **Dev/Test 데이터 (i>0)**: 이미 훈련 데이터에 등장한 단어만 추가
   - **목적**: OOV (Out-of-Vocabulary) 방지, 훈련 데이터 기반 어휘 구축

   **collections.Counter 개념**:
   - Python의 `collections.Counter`: 딕셔너리의 서브클래스
   - 자동으로 존재하지 않는 키를 0으로 초기화
   - `+=` 연산자로 빈도수 증가

#### 2.3.4 단어 사전 구축 및 필터링
**파일**: `MIND_corpus.py:104-111`

```python
# 3. word dictionary
word_counter_list = [[word, word_counter[word]] for word in word_counter]
word_counter_list.sort(key=lambda x: x[1], reverse=True)  # sort by word frequency
filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
for i, word in enumerate(filtered_word_counter_list):
    word_dict[word[0]] = i + 2
with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
    json.dump(word_dict, vocabulary_f)
```

**단계별 분석**:

1. **리스트 컴프리헨션**:
   ```python
   word_counter_list = [[word, word_counter[word]] for word in word_counter]
   ```
   - **입력**: `Counter({'the': 1000, 'lakers': 50, 'a': 800})`
   - **출력**: `[['the', 1000], ['lakers', 50], ['a', 800]]`

2. **람다 함수와 정렬**:
   ```python
   word_counter_list.sort(key=lambda x: x[1], reverse=True)
   ```
   - `lambda x: x[1]`: 익명 함수, 두 번째 요소(빈도수)를 정렬 키로 사용
   - `reverse=True`: 내림차순 정렬 (빈도 높은 순)
   - **결과**: `[['the', 1000], ['a', 800], ['lakers', 50]]`

3. **필터링**:
   ```python
   filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
   ```
   - `filter()`: 조건을 만족하는 요소만 선택
   - `config.word_threshold = 3`이면 3번 이상 등장한 단어만 유지
   - **목적**: 희귀 단어 제거, 노이즈 감소, 메모리 절약

4. **인덱스 할당**:
   ```python
   for i, word in enumerate(filtered_word_counter_list):
       word_dict[word[0]] = i + 2
   ```
   - **i + 2**: 왜 2부터 시작?
     - 0: `<PAD>` (패딩)
     - 1: `<UNK>` (Unknown, 어휘에 없는 단어)
     - 2부터: 실제 단어들
   - **결과 예시**: `{"<PAD>": 0, "<UNK>": 1, "the": 2, "a": 3, "lakers": 4}`

#### 2.3.5 GloVe 워드 임베딩 로딩
**파일**: `MIND_corpus.py:113-132`

```python
# 4. Glove word embedding
if config.word_embedding_dim == 300:
    glove = GloVe(name='840B', dim=300, cache='/home/user/jaesung/newsreclib/data/glove', max_vectors=10000000000)
else:
    glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='/home/user/jaesung/newsreclib/data/glove', max_vectors=10000000000)
glove_stoi = glove.stoi  # string to index
glove_vectors = glove.vectors
glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
```

**단계별 분석**:

1. **GloVe 로딩**:
   - **GloVe**: 단어의 의미를 벡터로 표현한 사전 학습 임베딩
   - `'840B'`: 840억 개 토큰으로 학습된 모델 (더 큰 어휘)
   - `'6B'`: 60억 개 토큰으로 학습된 모델

2. **주요 속성**:
   - `glove.stoi`: 딕셔너리, 단어 → 인덱스
     - 예: `{'the': 0, 'a': 1, 'lakers': 5000}`
   - `glove.vectors`: `[어휘크기, embedding_dim]` 텐서
     - 예: `torch.Size([400000, 300])`

3. **평균 벡터 계산**:
   ```python
   glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
   ```
   - `torch.mean()`: 평균 계산
   - `dim=0`: 첫 번째 차원(단어 차원)을 따라 평균
   - `keepdim=False`: 차원 유지 안 함
   - **입력 shape**: `[400000, 300]`
   - **출력 shape**: `[300]`
   - **목적**: GloVe에 없는 단어에 대한 초기화 기준

4. **임베딩 매트릭스 초기화**:
   ```python
   word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
   for word in word_dict:
       index = word_dict[word]
       if index != 0:  # PAD는 0벡터 유지
           if word in glove_stoi:
               word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
           else:
               random_vector = torch.zeros(config.word_embedding_dim)
               random_vector.normal_(mean=0, std=0.1)
               word_embedding_vectors[index, :] = random_vector + glove_mean_vector
   ```

   **로직 설명**:
   - **경우 1**: 단어가 GloVe에 존재
     - GloVe 벡터 그대로 사용
   - **경우 2**: 단어가 GloVe에 없음
     - 정규분포 노이즈 + GloVe 평균 벡터
     - `normal_(mean=0, std=0.1)`: 표준편차 0.1의 정규분포로 in-place 초기화
   - **경우 3**: PAD 토큰 (index==0)
     - 0 벡터 유지

5. **피클 저장**:
   ```python
   with open(word_embedding_file, 'wb') as word_embedding_f:
       pickle.dump(word_embedding_vectors, word_embedding_f)
   ```
   - `pickle`: Python 객체를 바이너리로 직렬화
   - `'wb'`: 쓰기 바이너리 모드
   - **저장 이유**: 매번 GloVe 로딩 시간 절약

#### 2.3.6 엔티티 임베딩 로딩
**파일**: `MIND_corpus.py:134-160`

```python
# 5. knowledge-graph entity dictionary & entity embedding & context embedding
entity_embedding_vectors = torch.zeros([len(entity_dict), config.entity_embedding_dim])
context_embedding_vectors = torch.zeros([len(entity_dict), config.context_embedding_dim])
for prefix in [config.train_root, config.dev_root, config.test_root]:
    with open(os.path.join(prefix, 'entity_embedding.vec'), 'r', encoding='utf-8') as entity_f:
        for line in entity_f:
            if len(line.strip()) > 0:
                terms = line.strip().split('\t')
                assert len(terms) == config.entity_embedding_dim + 1, 'entity embedding dim does not match'
                WikidataId = terms[0]
                if WikidataId in entity_dict:
                    entity_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
```

**단계별 분석**:

1. **엔티티란?**:
   - 뉴스 텍스트에서 인식된 개체 (사람, 장소, 조직 등)
   - Wikidata ID로 식별
   - 예: "Lakers" → "Q121903" (Wikidata ID)

2. **파일 형식**:
   ```
   Q121903    0.123  -0.456  0.789  ...  (100차원)
   Q1297    -0.234   0.567 -0.890  ...
   ```
   - 첫 열: Wikidata ID
   - 나머지: 임베딩 벡터 값들

3. **파싱 과정**:
   ```python
   terms = line.strip().split('\t')
   ```
   - `.strip()`: 앞뒤 공백 제거
   - `.split('\t')`: 탭으로 분할
   - **입력**: `"Q121903\t0.123\t-0.456\t...\n"`
   - **출력**: `['Q121903', '0.123', '-0.456', ...]`

4. **검증**:
   ```python
   assert len(terms) == config.entity_embedding_dim + 1
   ```
   - `assert`: 조건이 False면 AssertionError 발생
   - **검증 내용**: ID(1개) + 임베딩(100개) = 101개 확인

5. **텐서 변환**:
   ```python
   entity_embedding_vectors[entity_dict[WikidataId]] = torch.FloatTensor(list(map(float, terms[1:])))
   ```
   - `terms[1:]`: ID를 제외한 나머지 (임베딩 값들)
   - `map(float, ...)`: 문자열 → float 변환
   - `list(...)`: map 객체 → 리스트
   - `torch.FloatTensor(...)`: 리스트 → PyTorch 텐서

#### 2.3.7 유저 히스토리 그래프 생성
**파일**: `MIND_corpus.py:162-221`

```python
# 6. user history graph
category_num = len(category_dict)
graph_size = config.max_history_num + category_num  # graph size of |V_{n}|+|V_{p}|
```

**그래프 구조 설명**:
- **노드 타입 1**: 뉴스 노드 (최대 `max_history_num`개)
- **노드 타입 2**: 카테고리 프록시 노드 (`category_num`개)
- **목적**: 같은 카테고리의 뉴스들을 연결, 계층적 구조 표현

```python
user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=np.float32)
user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=bool)
user_history_category_indices = np.zeros([user_history_num, config.max_history_num], dtype=np.int64)
```

**데이터 구조**:
1. **user_history_graph**: `[사용자수, 그래프크기, 그래프크기]`
   - 인접 행렬 (Adjacency Matrix)
   - `[i, j, k]` = 1.0: 사용자 i의 노드 j와 k 사이에 엣지 존재

2. **user_history_category_mask**: `[사용자수, 카테고리수+1]`
   - 사용자가 읽은 카테고리 표시
   - `True`: 해당 카테고리 뉴스 읽음

3. **user_history_category_indices**: `[사용자수, max_history_num]`
   - 각 히스토리 뉴스의 카테고리 인덱스
   - 패딩 뉴스는 `category_num` (마지막 인덱스)

**그래프 구축 로직**:
```python
with open(os.path.join(prefix, 'behaviors_raw.tsv'), 'r', encoding='utf-8') as behaviors_f:
    for line_index, line in enumerate(behaviors_f):
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        if config.no_self_connection:
            history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
        else:
            history_graph = np.identity(graph_size, dtype=np.float32)
```

**Self-connection 개념**:
- `no_self_connection=False`: 각 노드가 자기 자신과 연결 (대각선 1)
- `no_self_connection=True`: 자기 자신과 연결 안 함 (대각선 0)
- `np.identity()`: 단위 행렬 생성

```python
history_news_ID = history.split(' ')
offset = max(0, len(history_news_ID) - config.max_history_num)
history_news_num = min(len(history_news_ID), config.max_history_num)
```

**오프셋 계산**:
- **목적**: 히스토리가 50개 넘으면 최근 50개만 사용
- **예시**:
  - 히스토리 70개: `offset=20`, 20~69번째 뉴스 사용
  - 히스토리 30개: `offset=0`, 0~29번째 뉴스 사용

```python
for i in range(history_news_num):
    category_index = news_category_dict[history_news_ID[i + offset]]
    history_category_mask[category_index] = 1
    history_category_indices[i] = category_index
    history_graph[i, config.max_history_num + category_index] = 1  # edge of E_{p}^{1}
    history_graph[config.max_history_num + category_index, i] = 1  # edge of E_{p}^{1}
```

**엣지 생성 로직**:
1. **뉴스 → 카테고리 엣지**:
   - `history_graph[i, max_history_num + category_index] = 1`
   - 뉴스 노드 i → 카테고리 노드 연결

2. **카테고리 → 뉴스 엣지**:
   - `history_graph[max_history_num + category_index, i] = 1`
   - 양방향 엣지 (무방향 그래프)

```python
for j in range(i + 1, history_news_num):
    _category_index = news_category_dict[history_news_ID[j + offset]]
    if category_index == _category_index:
        history_graph[i, j] = 1  # edge of E_{n}
        history_graph[j, i] = 1  # edge of E_{n}
    else:
        history_graph[max_history_num + category_index, max_history_num + _category_index] = 1  # edge of E_{p}^{2}
        history_graph[max_history_num + _category_index, max_history_num + category_index] = 1  # edge of E_{p}^{2}
```

**엣지 타입**:
1. **E_n (intra-cluster edge)**:
   - 같은 카테고리 뉴스끼리 연결
   - `category_index == _category_index`

2. **E_p^2 (inter-cluster edge)**:
   - 다른 카테고리 프록시 노드끼리 연결
   - `category_index != _category_index`

**인접 행렬 정규화**:
```python
if not config.no_adjacent_normalization:
    if config.gcn_normalization_type == 'asymmetric':
        # D^{-1}A
        D_inv = np.zeros([graph_size, graph_size], dtype=np.float32)
        np.fill_diagonal(D_inv, 1 / history_graph.sum(axis=1, keepdims=False))
        history_graph = np.matmul(D_inv, history_graph)
    else:
        # D^{-1/2}AD^{-1/2}
        D_inv_sqrt = np.zeros([graph_size, graph_size], dtype=np.float32)
        np.fill_diagonal(D_inv_sqrt, np.sqrt(1 / history_graph.sum(axis=1, keepdims=False)))
        history_graph = np.matmul(np.matmul(D_inv_sqrt, history_graph), D_inv_sqrt)
```

**정규화 이유**:
- GCN (Graph Convolutional Network)에서 필수
- 차수(degree)가 다른 노드들의 영향력을 균등하게 조정

**Asymmetric vs Symmetric**:
1. **Asymmetric (D^{-1}A)**:
   - 각 노드의 출력을 이웃 수로 나눔
   - 행 단위 정규화

2. **Symmetric (D^{-1/2}AD^{-1/2})**:
   - 양쪽 노드의 차수를 모두 고려
   - 더 균형잡힌 정규화

**수식 설명**:
- **D**: 차수 행렬 (Degree matrix), 대각 행렬
  - `D[i, i]` = 노드 i의 차수 (연결된 엣지 수)
- **A**: 인접 행렬 (Adjacency matrix)
- `history_graph.sum(axis=1)`: 각 행의 합 = 각 노드의 차수

---

## 3. 데이터셋 클래스와 데이터 로더

### 3.1 파일 위치: `MIND_dataset.py`

### 3.2 훈련 데이터셋 클래스

#### 3.2.1 클래스 정의
**파일**: `MIND_dataset.py:9-26`

```python
class MIND_Train_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus):
        self.negative_sample_num = corpus.negative_sample_num
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text = corpus.news_title_text
        ...
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)
```

**클래스 상속**:
- `data.Dataset`: PyTorch의 `torch.utils.data.Dataset` 추상 클래스
- **반드시 구현해야 하는 메서드**:
  1. `__len__()`: 데이터셋 크기 반환
  2. `__getitem__(index)`: 인덱스에 해당하는 샘플 반환

**멤버 변수**:
- `self.train_behaviors`: 리스트의 리스트
  ```python
  [
    [user_ID, [history_news_IDs], history_mask, positive_news_ID, [negative_news_IDs], behavior_index],
    ...
  ]
  ```
- `self.train_samples`: 2D 리스트, `[샘플수][1+neg_num]`
  - 첫 번째: positive 샘플
  - 나머지: negative 샘플들

#### 3.2.2 Negative Sampling
**파일**: `MIND_dataset.py:27-47`

```python
def negative_sampling(self, rank=None):
    print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
    start_time = time.time()
    for i, train_behavior in enumerate(self.train_behaviors):
        self.train_samples[i][0] = train_behavior[3]  # positive sample
        negative_samples = train_behavior[4]
        news_num = len(negative_samples)
        if news_num <= self.negative_sample_num:
            for j in range(self.negative_sample_num):
                self.train_samples[i][j + 1] = negative_samples[j % news_num]
        else:
            used_negative_samples = set()
            for j in range(self.negative_sample_num):
                while True:
                    k = randint(0, news_num)
                    if k not in used_negative_samples:
                        self.train_samples[i][j + 1] = negative_samples[k]
                        used_negative_samples.add(k)
                        break
```

**Negative Sampling이란?**:
- **목적**: 클릭하지 않은 뉴스(negative)와 클릭한 뉴스(positive)를 함께 학습
- **왜 필요?**: 추천 시스템에서 positive만 학습하면 모든 뉴스를 추천하는 trivial solution 발생

**로직 분석**:

1. **Positive 샘플**:
   ```python
   self.train_samples[i][0] = train_behavior[3]
   ```
   - 사용자가 실제로 클릭한 뉴스

2. **경우 1: Negative 샘플이 부족한 경우** (`news_num <= negative_sample_num`):
   ```python
   for j in range(self.negative_sample_num):
       self.train_samples[i][j + 1] = negative_samples[j % news_num]
   ```
   - 모듈로 연산(`%`)으로 반복 선택
   - **예시**: negative 2개, 필요 4개 → [0, 1, 0, 1]

3. **경우 2: Negative 샘플이 충분한 경우** (`news_num > negative_sample_num`):
   ```python
   used_negative_samples = set()
   while True:
       k = randint(0, news_num)
       if k not in used_negative_samples:
           self.train_samples[i][j + 1] = negative_samples[k]
           used_negative_samples.add(k)
           break
   ```
   - `set()`: 중복 방지
   - `randint(0, news_num)`: 0부터 news_num-1까지 랜덤 정수
   - **목적**: 중복 없이 랜덤하게 negative 샘플 선택

**시간 측정**:
```python
start_time = time.time()
# ... 샘플링 로직 ...
end_time = time.time()
print('%sEnd negative sampling, used time : %.3fs' % (..., end_time - start_time))
```
- `time.time()`: Unix epoch 이후 초 (float)
- 시간 차이로 소요 시간 계산

#### 3.2.3 데이터 인덱싱
**파일**: `MIND_dataset.py:70-76`

```python
def __getitem__(self, index):
    train_behavior = self.train_behaviors[index]
    history_index = train_behavior[1]
    sample_index = self.train_samples[index]
    behavior_index = train_behavior[5]
    return train_behavior[0], self.news_category[history_index], ...
```

**반환 형태 분석**:

입력: `index = 42`

1. **User 정보**:
   - `train_behavior[0]`: user_ID (스칼라)

2. **User History**:
   - `history_index = train_behavior[1]`: `[max_history_num]` 크기의 뉴스 ID 리스트
   - `self.news_category[history_index]`: NumPy 고급 인덱싱
     - **입력**: `history_index = [5, 10, 15, 0, 0, ...]` (0은 패딩)
     - **출력**: `[2, 3, 2, 0, 0, ...]` (각 뉴스의 카테고리)

3. **Candidate News**:
   - `sample_index = self.train_samples[index]`: `[1 + negative_sample_num]`
   - 첫 번째: positive, 나머지: negative

**NumPy 고급 인덱싱**:
```python
arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]
result = arr[indices]  # [10, 30, 50]
```
- 정수 리스트로 인덱싱하면 해당 위치의 값들을 추출
- 2D 배열에도 적용 가능

#### 3.2.4 DataLoader 통합
**파일**: `trainer.py:78`

```python
train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
```

**PyTorch DataLoader 파라미터**:

1. **batch_size**:
   - 한 번에 로드할 샘플 수
   - GPU 메모리에 따라 조정

2. **shuffle=True**:
   - 매 epoch마다 데이터 순서 섞음
   - **이유**: 학습 안정성, overfitting 방지

3. **num_workers**:
   - 데이터 로딩용 서브프로세스 수
   - `batch_size // 16`: 경험적 휴리스틱
   - **0**: 메인 프로세스에서 로딩 (느림)
   - **>0**: 멀티프로세싱으로 병렬 로딩 (빠름)

4. **pin_memory=True**:
   - CPU 메모리를 고정(pinned)
   - **효과**: CPU→GPU 데이터 전송 속도 향상
   - **주의**: CPU 메모리 부족 시 False

**데이터 로딩 흐름**:
```
DataLoader
  ↓ (병렬)
num_workers개 프로세스가 __getitem__() 호출
  ↓ (배치 구성)
collate_fn으로 리스트 → 텐서 변환 (기본 동작)
  ↓
pin_memory로 고정 메모리 할당
  ↓
메인 프로세스에 반환
```

### 3.3 검증/테스트 데이터셋 클래스

**파일**: `MIND_dataset.py:82-129`

```python
class MIND_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        ...
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
```

**훈련 데이터셋과의 차이점**:

1. **Negative Sampling 없음**:
   - 모든 candidate 뉴스를 평가
   - 랭킹 성능 측정이 목적

2. **샘플 구조**:
   ```python
   [user_ID, [history], history_mask, candidate_news_ID, behavior_index]
   ```
   - 각 샘플은 하나의 candidate 뉴스만 포함
   - impression 단위가 아닌 개별 뉴스 단위

3. **평가 방식**:
   - 같은 impression의 모든 뉴스에 대해 점수 계산
   - 점수 순위로 AUC, MRR, nDCG 계산

---

## 4. NAML 모델 아키텍처

### 4.1 전체 구조

```
입력
  ├─ User History (뉴스 제목, 내용, 카테고리)
  └─ Candidate News (뉴스 제목, 내용, 카테고리)
       ↓
News Encoder (NAML)
  ├─ Title CNN
  ├─ Content CNN
  ├─ Category Embedding
  └─ SubCategory Embedding
       ↓ Multi-view Attention
  News Representation
       ↓
User Encoder
  History Aggregation
       ↓
  User Representation
       ↓
Click Predictor (Dot Product)
       ↓
  Click Score
```

### 4.2 NAML News Encoder

**파일**: `newsEncoders.py:281-329`

#### 4.2.1 클래스 정의
```python
class NAML(NewsEncoder):
    def __init__(self, config: Config):
        super(NAML, self).__init__(config)
        self.max_title_length = config.max_title_length
        self.max_content_length = config.max_abstract_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.news_embedding_dim = config.cnn_kernel_num
```

**클래스 상속 관계**:
```
nn.Module (PyTorch 기본)
  ↓
NewsEncoder (newsEncoders.py:11)
  ↓
NAML (newsEncoders.py:281)
```

**NewsEncoder 부모 클래스**:
- **위치**: `newsEncoders.py:11-54`
- **제공 기능**:
  1. `word_embedding`: GloVe 워드 임베딩
  2. `category_embedding`: 카테고리 임베딩
  3. `subCategory_embedding`: 서브카테고리 임베딩
  4. `feature_fusion()`: 카테고리 정보 통합 메서드

**nn.Module 개념**:
- PyTorch의 모든 신경망 모듈의 베이스 클래스
- **핵심 기능**:
  1. 파라미터 자동 추적: `self.parameters()` 호출 가능
  2. GPU 이동: `.cuda()` 메서드 지원
  3. 훈련/평가 모드: `.train()`, `.eval()` 전환

#### 4.2.2 서브모듈 초기화
```python
self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
```

**Conv1D 레이어**:
- **목적**: 단어 시퀀스에서 로컬 패턴 추출
- **파라미터**:
  - `in_channels=word_embedding_dim`: 입력 채널 (300)
  - `out_channels=cnn_kernel_num`: 출력 채널 (400)
  - `kernel_size=window_size`: 커널 크기 (3)

**1D Convolution 동작 원리**:
```
입력: [batch, channels=300, length=32]
  ↓
1D Conv (kernel=3)
  ↓
출력: [batch, channels=400, length=32]
```
- 3개 연속 단어의 패턴을 학습
- 400개의 서로 다른 필터로 다양한 패턴 캡처

**Attention 레이어**:
- **목적**: 중요한 단어에 가중치 부여
- **파라미터**:
  - `hidden_dim=cnn_kernel_num`: 입력 차원
  - `attention_dim`: 어텐션 계산용 중간 차원

#### 4.2.3 Multi-view Attention
```python
self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
self.subCategory_affine = nn.Linear(config.subCategory_embedding_dim, config.cnn_kernel_num, bias=True)
self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
```

**Affine Transformation**:
- `nn.Linear`: 선형 변환 레이어 (Fully Connected)
- **수식**: `y = Wx + b`
  - `W`: `[out_features, in_features]` weight 행렬
  - `b`: `[out_features]` bias 벡터

**Multi-view Attention 구조**:
```
View 1: Title → CNN → Attention → [batch, news_num, 400]
View 2: Content → CNN → Attention → [batch, news_num, 400]
View 3: Category → Embedding → Linear → [batch, news_num, 400]
View 4: SubCategory → Embedding → Linear → [batch, news_num, 400]
  ↓ Stack
[batch, news_num, 4, 400]
  ↓ Attention
[batch, news_num, 400]
```

#### 4.2.4 Forward 메서드
**파일**: `newsEncoders.py:309-329`

```python
def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
    batch_size = title_text.size(0)
    news_num = title_text.size(1)
    batch_news_num = batch_size * news_num
```

**입력 텐서 형태**:
- `title_text`: `[batch_size, news_num, max_title_length]`
  - 예: `[64, 5, 32]` = 64개 배치, 5개 뉴스, 32 단어
- `title_mask`: `[batch_size, news_num, max_title_length]`
  - Boolean 텐서, True: 실제 단어, False: 패딩
- `category`: `[batch_size, news_num]`
  - 정수 텐서, 카테고리 ID

**Step 1: Word Embedding**
```python
# 1. word embedding
title_w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_title_length, self.word_embedding_dim])
content_w = self.dropout(self.word_embedding(content_text)).view([batch_news_num, self.max_content_length, self.word_embedding_dim])
```

**연산 분석**:

1. `self.word_embedding(title_text)`:
   - **입력**: `[64, 5, 32]` (정수 인덱스)
   - **임베딩 룩업**: 각 인덱스 → 300차원 벡터
   - **출력**: `[64, 5, 32, 300]`

2. `.view([batch_news_num, ...])`:
   - **목적**: 배치와 뉴스 차원을 하나로 합침
   - **입력**: `[64, 5, 32, 300]`
   - **출력**: `[320, 32, 300]` (64*5=320)
   - **이유**: CNN은 3D 텐서만 처리 가능

3. `self.dropout(...)`:
   - 랜덤하게 일부 값을 0으로 설정
   - **확률**: `config.dropout_rate` (예: 0.2)
   - **목적**: Overfitting 방지, 일반화 향상

**PyTorch Dropout 동작**:
```python
# dropout_rate = 0.2
input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
# 훈련 시
output = dropout(input)  # [1.25, 0.0, 3.75, 0.0, 6.25] (20% 확률로 0, 나머지는 1/(1-0.2) 배)
# 평가 시
output = dropout(input)  # [1.0, 2.0, 3.0, 4.0, 5.0] (그대로)
```
- 훈련 시: 드롭아웃 적용 + 스케일링
- 평가 시: 아무 작업 안 함

**Step 2: CNN Encoding**
```python
# 2. CNN encoding
title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))
content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))
```

**Permute 이유**:
```python
title_w.shape  # [320, 32, 300]
title_w.permute(0, 2, 1).shape  # [320, 300, 32]
```
- PyTorch Conv1d: `[batch, channels, length]` 형태 필요
- 원래: `[batch, length, channels]` (단어 시퀀스 관점)
- 변환 후: `[batch, channels, length]` (Conv1d 관점)

**Conv1D 연산**:
```python
self.title_conv(...)  # [320, 300, 32] → [320, 400, 32]
```
- 입력 채널: 300 (word embedding dim)
- 출력 채널: 400 (cnn_kernel_num)
- 시퀀스 길이: 32 유지 (padding='same' 효과)

**다시 Permute**:
```python
.permute(0, 2, 1)  # [320, 400, 32] → [320, 32, 400]
```
- Attention 레이어는 `[batch, length, features]` 형태 필요

**Step 3: Attention Layer**
```python
# 3. attention layer
title_representation = self.title_attention(title_c).view([batch_size, news_num, self.cnn_kernel_num])
content_representation = self.content_attention(content_c).view([batch_size, news_num, self.cnn_kernel_num])
```

**Attention 메커니즘**:
```python
# title_c: [320, 32, 400]
# Attention 계산
scores = tanh(W * title_c + b)  # [320, 32, attention_dim]
weights = softmax(V * scores)    # [320, 32, 1]
output = sum(title_c * weights)  # [320, 400]
```

**Softmax 함수**:
```python
def softmax(x):
    exp_x = torch.exp(x - x.max())  # 수치 안정성
    return exp_x / exp_x.sum()
```
- **입력**: `[2.0, 1.0, 0.1]`
- **출력**: `[0.659, 0.242, 0.099]` (합=1.0)
- **의미**: 가장 큰 값에 높은 가중치

**View 연산**:
```python
.view([batch_size, news_num, self.cnn_kernel_num])
# [320, 400] → [64, 5, 400]
```
- 다시 배치와 뉴스 차원 분리

**Step 4: Category Encoding**
```python
# 4. category and subCategory encoding
category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)
subCategory_representation = F.relu(self.subCategory_affine(self.subCategory_embedding(subCategory)), inplace=True)
```

**연산 체인**:
```python
category  # [64, 5] (정수 인덱스)
  ↓
self.category_embedding(...)  # [64, 5, 50] (임베딩 룩업)
  ↓
self.category_affine(...)     # [64, 5, 400] (선형 변환)
  ↓
F.relu(..., inplace=True)     # [64, 5, 400] (ReLU 활성화)
```

**ReLU 활성화 함수**:
```python
def relu(x):
    return max(0, x)
```
- 음수 → 0
- 양수 → 그대로
- **inplace=True**: 메모리 절약, 원본 텐서 수정

**Step 5: Multi-view Attention**
```python
# 5. multi-view attention
feature = torch.stack([title_representation, content_representation, category_representation, subCategory_representation], dim=2)
alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)
news_representation = (feature * alpha).sum(dim=2, keepdim=False)
```

**Stack 연산**:
```python
torch.stack([A, B, C, D], dim=2)
# A, B, C, D: [64, 5, 400]
# 출력: [64, 5, 4, 400]
```
- 새로운 차원 생성 (dim=2)
- 4개 뷰를 쌓음

**Attention 계산**:
```python
self.affine1(feature)  # [64, 5, 4, 400] → [64, 5, 4, 200]
torch.tanh(...)        # [-1, 1] 범위로 압축
self.affine2(...)      # [64, 5, 4, 200] → [64, 5, 4, 1]
F.softmax(..., dim=2)  # dim=2 방향으로 합=1
```

**Softmax의 dim 파라미터**:
```python
# dim=2: 4개 뷰의 가중치 합 = 1
alpha[batch, news, :, :]  # [view1, view2, view3, view4] 가중치가 [0.3, 0.2, 0.1, 0.4]
```

**가중 합**:
```python
(feature * alpha)  # 브로드캐스팅: [64, 5, 4, 400] * [64, 5, 4, 1]
.sum(dim=2, keepdim=False)  # [64, 5, 400]
```
- `keepdim=False`: 합산한 차원 제거
- 최종: 4개 뷰의 가중 평균

**반환**:
```python
return news_representation  # [64, 5, 400]
```
- 64개 배치, 5개 뉴스, 400차원 표현

### 4.3 User Encoder (ATT 예시)

**파일**: `userEncoders.py:176-191`

```python
class ATT(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(ATT, self).__init__(news_encoder, config)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)
```

**클래스 상속 관계**:
```
nn.Module
  ↓
UserEncoder (userEncoders.py:12)
  ↓
ATT (userEncoders.py:176)
```

**UserEncoder 부모 클래스**:
- **위치**: `userEncoders.py:12-39`
- **멤버 변수**:
  - `self.news_encoder`: News Encoder 참조
  - `self.news_embedding_dim`: 뉴스 표현 차원
  - `self.auxiliary_loss`: 보조 손실 (일부 모델만 사용)

#### Forward 메서드
```python
def forward(self, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_category, user_subCategory, \
            user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
    news_num = candidate_news_representation.size(1)
    history_embedding = self.news_encoder(user_title_text, user_title_mask, user_title_entity, \
                                          user_content_text, user_content_mask, user_content_entity, \
                                          user_category, user_subCategory, user_embedding)
    user_representation = self.attention(history_embedding).unsqueeze(dim=1).expand(-1, news_num, -1)
    return user_representation
```

**입력 텐서 형태**:
- `user_title_text`: `[batch_size, max_history_num, max_title_length]`
  - 예: `[64, 50, 32]` = 64명, 50개 히스토리, 32 단어
- `candidate_news_representation`: `[batch_size, news_num, news_embedding_dim]`
  - 예: `[64, 5, 400]` = 64명, 5개 후보 뉴스, 400차원

**Step 1: History Encoding**
```python
history_embedding = self.news_encoder(...)  # [64, 50, 400]
```
- 각 히스토리 뉴스를 News Encoder로 인코딩
- 사용자의 과거 읽기 이력 표현

**Step 2: Attention Aggregation**
```python
self.attention(history_embedding)  # [64, 50, 400] → [64, 400]
```
- 50개 히스토리를 하나의 벡터로 집계
- 중요한 히스토리에 가중치 부여

**Step 3: Expansion**
```python
.unsqueeze(dim=1)  # [64, 400] → [64, 1, 400]
.expand(-1, news_num, -1)  # [64, 1, 400] → [64, 5, 400]
```

**Unsqueeze vs Expand**:
- `unsqueeze(dim=1)`: 크기 1인 차원 추가
- `expand(-1, 5, -1)`: 해당 차원을 5로 반복
  - `-1`: 해당 차원 크기 유지
  - **주의**: 메모리 복사 없음, view만 변경

**왜 Expand?**:
- 각 후보 뉴스마다 동일한 사용자 표현 필요
- `[64, 5, 400]` 형태로 맞춤

### 4.4 Click Predictor

**파일**: `model.py:126-133`

```python
if self.click_predictor == 'dot_product':
    logits = (user_representation * news_representation).sum(dim=2)
elif self.click_predictor == 'mlp':
    context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, news_representation], dim=2)), inplace=True))
    logits = self.out(context).squeeze(dim=2)
```

**Dot Product 방식**:
```python
(user_representation * news_representation)  # [64, 5, 400] * [64, 5, 400] = [64, 5, 400]
.sum(dim=2)  # [64, 5]
```
- 요소별 곱셈 후 합산
- 벡터 내적 (Dot Product)
- **수식**: `score = u · n = Σ(u_i * n_i)`
- **의미**: 사용자와 뉴스의 유사도

**MLP 방식**:
```python
torch.cat([user_representation, news_representation], dim=2)  # [64, 5, 400] + [64, 5, 400] = [64, 5, 800]
self.mlp(...)  # [64, 5, 800] → [64, 5, 200]
F.relu(...)    # 비선형 활성화
self.out(...)  # [64, 5, 200] → [64, 5, 1]
.squeeze(dim=2)  # [64, 5]
```
- 연결(concatenation) 후 MLP
- 더 복잡한 상호작용 학습 가능

---

## 5. 훈련 과정 (Forward & Backward)

### 5.1 Trainer 클래스 초기화

**파일**: `trainer.py:19-62`

```python
class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
        self.model = model
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
```

**멤버 변수 설명**:

1. **self.model**: 훈련할 모델 인스턴스

2. **self.loss**: 손실 함수 선택
   - **Negative Log Softmax**: 다중 클래스 분류 (positive vs negatives)
   - **Negative Log Sigmoid**: 이진 분류 (각 뉴스 독립적으로)

3. **self.optimizer**: Adam 옵티마이저
   ```python
   optim.Adam(parameters, lr=1e-4, weight_decay=0)
   ```
   - `parameters`: 업데이트할 파라미터
   - `lr`: 학습률 (learning rate)
   - `weight_decay`: L2 정규화 계수

**filter(lambda p: p.requires_grad, ...)**:
```python
for p in self.model.parameters():
    if p.requires_grad:
        # 이 파라미터만 옵티마이저에 포함
```
- `requires_grad=True`: 그래디언트 계산 필요
- `requires_grad=False`: 고정된 파라미터 (예: 사전 학습 임베딩)

**Adam 옵티마이저**:
- **Adam**: Adaptive Moment Estimation
- **특징**:
  1. 각 파라미터마다 적응적 학습률
  2. 모멘텀 사용 (관성)
  3. RMSprop + Momentum 결합
- **수식**:
  ```
  m_t = β1 * m_{t-1} + (1 - β1) * g_t  (1차 모멘트)
  v_t = β2 * v_{t-1} + (1 - β2) * g_t^2  (2차 모멘트)
  θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
  ```

### 5.2 훈련 루프

**파일**: `trainer.py:74-185`

```python
def train(self):
    model = self.model
    for e in tqdm(range(1, self.epoch + 1)):
        self.train_dataset.negative_sampling()
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
        model.train()
        epoch_loss = 0
```

**Epoch 루프**:
- `range(1, self.epoch + 1)`: 1부터 epoch까지 (1-based)
- `tqdm()`: 진행률 표시 (Progress bar)

**model.train()**:
```python
model.train()  # 훈련 모드
# vs
model.eval()   # 평가 모드
```
- **훈련 모드**:
  - Dropout 활성화
  - BatchNorm 통계 업데이트
  - `requires_grad=True` 파라미터 그래디언트 계산
- **평가 모드**:
  - Dropout 비활성화
  - BatchNorm 저장된 통계 사용
  - 일반적으로 `torch.no_grad()`와 함께 사용

### 5.3 배치 처리 루프

**파일**: `trainer.py:81-103`

```python
for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
    news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) in train_dataloader:
    user_ID = user_ID.cuda(non_blocking=True)
    user_category = user_category.cuda(non_blocking=True)
    ...
```

**데이터 로더 언패킹**:
- DataLoader가 배치를 튜플로 반환
- 각 변수에 자동 할당

**GPU 전송**:
```python
user_ID = user_ID.cuda(non_blocking=True)
```
- `.cuda()`: CPU 텐서 → GPU 텐서
- `non_blocking=True`: 비동기 전송
  - CPU와 GPU 작업 병렬화
  - `pin_memory=True`와 함께 사용 시 효과적

**텐서 형태 (주석 참고)**:
```python
user_ID = user_ID.cuda(non_blocking=True)  # [batch_size]
user_category = user_category.cuda(non_blocking=True)  # [batch_size, max_history_num]
user_title_text = user_title_text.cuda(non_blocking=True)  # [batch_size, max_history_num, max_title_length]
news_category = news_category.cuda(non_blocking=True)  # [batch_size, 1 + negative_sample_num]
news_title_text = news_title_text.cuda(non_blocking=True)  # [batch_size, 1 + negative_sample_num, max_title_length]
```

### 5.4 Forward Pass

**파일**: `trainer.py:105-106`

```python
logits = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
               news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity)
```

**모델 호출**:
```python
model(...)
# 실제로는
model.forward(...)
```
- Python의 `__call__` 매직 메서드
- `nn.Module`에서 정의됨

**Forward 경로**:
```
Model.forward()
  ├─ News Encoder (NAML)
  │   ├─ Title CNN + Attention
  │   ├─ Content CNN + Attention
  │   ├─ Category Affine
  │   ├─ SubCategory Affine
  │   └─ Multi-view Attention
  ├─ User Encoder (ATT)
  │   ├─ History News Encoding (재사용 News Encoder)
  │   └─ History Attention Aggregation
  └─ Click Predictor (Dot Product)
      └─ Dot Product
```

**logits 출력**:
```python
logits.shape  # [batch_size, 1 + negative_sample_num]
# 예: [64, 5] = 64개 샘플, 5개 뉴스 (1 positive + 4 negative)
```

### 5.5 Loss 계산

**파일**: `trainer.py:64-72`

```python
def negative_log_softmax(self, logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(self, logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss
```

**Negative Log Softmax 분석**:

1. **Softmax 계산**:
   ```python
   torch.log_softmax(logits, dim=1)
   ```
   - **입력**: `logits = [[2.0, 1.0, 0.5, 0.3, 0.1]]`
   - **Softmax**: `exp(x_i) / Σexp(x_j)`
     - `[2.0, 1.0, 0.5, 0.3, 0.1]`
     - → `[0.526, 0.194, 0.117, 0.096, 0.087]`
   - **Log Softmax**: `log(exp(x_i) / Σexp(x_j)) = x_i - log(Σexp(x_j))`
     - 수치 안정성 향상

2. **Select 연산**:
   ```python
   .select(dim=1, index=0)
   ```
   - dim=1의 0번째 인덱스 선택 (positive 뉴스)
   - `[64, 5]` → `[64]`

3. **Negative Log Likelihood**:
   ```python
   -torch.log_softmax(...).select(...)
   ```
   - **목표**: positive 뉴스의 확률 최대화
   - **수식**: `Loss = -log(P(positive))`
   - P가 클수록 Loss 작음

4. **평균**:
   ```python
   .mean()
   ```
   - 배치 평균 손실
   - `[64]` → 스칼라

**왜 Negative Log?**:
- 확률 P를 최대화 = -log(P)를 최소화
- Log: 수치 안정성, 곱셈을 덧셈으로 변환
- Negative: 최대화 → 최소화 문제로 변환

**Negative Log Sigmoid 분석**:

1. **Positive 뉴스**:
   ```python
   positive_sigmoid = torch.sigmoid(logits[:, 0])
   ```
   - 첫 번째 뉴스 (positive)의 시그모이드
   - 클릭 확률로 해석

2. **Negative 뉴스**:
   ```python
   negative_sigmoid = torch.sigmoid(-logits[:, 1:])
   ```
   - 나머지 뉴스 (negative)
   - `-logits`: 클릭 안 할 확률로 변환

3. **Clamp 연산**:
   ```python
   torch.clamp(..., min=1e-15, max=1)
   ```
   - `min=1e-15`: log(0) 방지 (수치 안정성)
   - `max=1`: 확률 상한

4. **Loss 합산**:
   ```python
   -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
   ```
   - 모든 예측의 로그 확률 합
   - `.numel()`: 텐서의 총 요소 수 (batch_size * (1 + neg_num))
   - 정규화: 평균 손실

**파일**: `trainer.py:108-114`

```python
loss = self.loss(logits)
if model.news_encoder.auxiliary_loss is not None:
    news_auxiliary_loss = model.news_encoder.auxiliary_loss.mean()
    loss += news_auxiliary_loss
if model.user_encoder.auxiliary_loss is not None:
    user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
    loss += user_encoder_auxiliary_loss
```

**Auxiliary Loss**:
- 일부 모델 (예: DAE)에서 사용
- 보조 학습 목표 추가
- 예: DAE의 reconstruction loss

### 5.6 Backward Pass

**파일**: `trainer.py:115-120`

```python
epoch_loss += float(loss) * user_ID.size(0)
self.optimizer.zero_grad()
loss.backward()
if self.gradient_clip_norm > 0:
    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
self.optimizer.step()
```

**Step 1: Loss 누적**
```python
epoch_loss += float(loss) * user_ID.size(0)
```
- `float(loss)`: 텐서 → Python float
- `user_ID.size(0)`: 배치 크기
- **이유**: 배치 평균 손실 × 배치 크기 = 배치 총 손실
- Epoch 총 손실 계산용

**Step 2: 그래디언트 초기화**
```python
self.optimizer.zero_grad()
```
- 이전 배치의 그래디언트 제거
- **필수**: PyTorch는 그래디언트를 누적함
- 안 하면: 그래디언트가 계속 쌓임

**Step 3: 역전파**
```python
loss.backward()
```
- **Backpropagation**: Chain Rule로 모든 파라미터의 그래디언트 계산
- **수식**: `∂Loss/∂θ`
- **PyTorch Autograd**:
  - 모든 연산을 computational graph로 기록
  - `.backward()` 호출 시 역방향으로 그래디언트 계산

**Autograd 예시**:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = x^2
z = 3 * y   # z = 3y = 3x^2
z.backward()
print(x.grad)  # dz/dx = 6x = 12.0
```

**Step 4: Gradient Clipping**
```python
if self.gradient_clip_norm > 0:
    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
```
- **목적**: Exploding Gradient 방지
- **방법**: 그래디언트 L2 norm을 최대값으로 제한
- **수식**:
  ```
  if ||g|| > max_norm:
      g = g * (max_norm / ||g||)
  ```

**왜 필요?**:
- RNN, LSTM 등에서 그래디언트 폭발 발생 가능
- 학습 불안정 방지

**Step 5: 파라미터 업데이트**
```python
self.optimizer.step()
```
- 계산된 그래디언트로 파라미터 업데이트
- **Adam 수식**:
  ```
  θ = θ - α * m / (√v + ε)
  ```
  - `θ`: 파라미터
  - `α`: 학습률
  - `m`: 1차 모멘트 (평균)
  - `v`: 2차 모멘트 (분산)

**전체 흐름**:
```
1. forward() → logits
2. loss 계산
3. zero_grad() → 그래디언트 초기화
4. backward() → 그래디언트 계산
5. clip_grad_norm_() → 그래디언트 제한
6. step() → 파라미터 업데이트
```

### 5.7 검증 (Validation)

**파일**: `trainer.py:124-130`

```python
# validation
auc, mrr, ndcg5, ndcg10 = compute_scores(model, self.mind_corpus, self.batch_size * 3 // 2, 'dev', self.dev_res_dir + '/' + model.model_name + '-' + str(e) + '.txt', self._dataset)
self.auc_results.append(auc)
self.mrr_results.append(mrr)
self.ndcg5_results.append(ndcg5)
self.ndcg10_results.append(ndcg10)
print('Epoch %d : dev done\nDev criterions' % e)
print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
```

**매 Epoch마다 검증**:
- 훈련 후 즉시 평가
- Overfitting 모니터링
- Early Stopping 판단

**배치 크기 증가**:
```python
self.batch_size * 3 // 2
```
- 검증 시: 그래디언트 계산 불필요 → 메모리 절약
- 더 큰 배치 → 더 빠른 평가

### 5.8 Early Stopping

**파일**: `trainer.py:132-185`

```python
if self.dev_criterion == 'auc':
    if auc >= self.best_dev_auc:
        self.best_dev_auc = auc
        self.best_dev_epoch = e
        with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
            result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
        self.epoch_not_increase = 0
    else:
        self.epoch_not_increase += 1
```

**Early Stopping 로직**:

1. **성능 향상 시**:
   - 최고 성능 갱신
   - 카운터 초기화
   - 결과 파일 저장

2. **성능 향상 없을 시**:
   - 카운터 증가
   - `epoch_not_increase` 카운팅

3. **종료 조건**:
   ```python
   if self.epoch_not_increase == self.early_stopping_epoch:
       break
   ```
   - 예: `early_stopping_epoch=5`
   - 5 epoch 동안 향상 없으면 종료

**왜 Early Stopping?**:
- Overfitting 방지
- 훈련 시간 절약
- 최적 시점 자동 선택

### 5.9 모델 저장

**파일**: `trainer.py:182-183`

```python
if self.epoch_not_increase == 0:
    torch.save({model.model_name: model.state_dict()}, self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch))
```

**state_dict()**:
```python
model.state_dict()
# 출력 예시:
{
    'word_embedding.weight': tensor(...),
    'title_conv.weight': tensor(...),
    'title_conv.bias': tensor(...),
    ...
}
```
- 모든 학습 가능한 파라미터의 딕셔너리
- 모델 구조는 저장 안 함 (파라미터만)

**torch.save()**:
```python
torch.save(obj, path)
```
- Python 객체를 파일로 직렬화
- 내부적으로 pickle 사용

**저장 시점**:
- 성능이 향상될 때마다 (`epoch_not_increase == 0`)
- 이전 최고 모델 덮어쓰기

---

## 6. 평가 과정

### 6.1 평가 함수

**파일**: `util.py:10-68`

```python
def compute_scores(model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int, mode: str, result_file: str, dataset: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    dataloader = DataLoader(MIND_DevTest_Dataset(mind_corpus, mode), batch_size=batch_size, shuffle=False, num_workers=batch_size // 16, pin_memory=True)
    indices = (mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    torch.cuda.empty_cache()
    model.eval()
```

**평가 준비**:

1. **DataLoader 생성**:
   - `shuffle=False`: 순서 유지 필수 (인덱스 매칭)
   - `MIND_DevTest_Dataset`: Negative sampling 없음

2. **Scores 텐서**:
   ```python
   scores = torch.zeros([len(indices)]).cuda()
   ```
   - 모든 candidate 뉴스의 점수 저장
   - GPU에 미리 할당

3. **캐시 비우기**:
   ```python
   torch.cuda.empty_cache()
   ```
   - 사용하지 않는 GPU 메모리 해제
   - 평가용 메모리 확보

4. **평가 모드**:
   ```python
   model.eval()
   ```
   - Dropout 비활성화
   - BatchNorm frozen

### 6.2 점수 계산

**파일**: `util.py:18-51`

```python
with torch.no_grad():
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) in dataloader:
        ...
        news_category = news_category.unsqueeze(dim=1)
        news_subCategory = news_subCategory.unsqueeze(dim=1)
        news_title_text = news_title_text.unsqueeze(dim=1)
        ...
        scores[index: index+batch_size] = model(...).squeeze(dim=1)
        index += batch_size
```

**torch.no_grad()**:
```python
with torch.no_grad():
    # 이 블록 안에서는 autograd 비활성화
    output = model(input)
```
- **효과**:
  1. 그래디언트 계산 안 함 → 메모리 절약
  2. 연산 속도 향상
- **주의**: 훈련 시에는 절대 사용 금지

**Unsqueeze 이유**:
```python
news_category.shape  # [batch_size]
news_category.unsqueeze(dim=1).shape  # [batch_size, 1]
```
- 모델은 `[batch_size, news_num]` 형태 기대
- 평가 시: 뉴스 하나씩 평가 → `news_num=1`

**점수 저장**:
```python
scores[index: index+batch_size] = model(...).squeeze(dim=1)
```
- `model(...)`: `[batch_size, 1]`
- `.squeeze(dim=1)`: `[batch_size]`
- 슬라이싱으로 누적 저장

### 6.3 점수 그룹화

**파일**: `util.py:52-55`

```python
scores = scores.tolist()
sub_scores = [[] for _ in range(indices[-1] + 1)]
for i, index in enumerate(indices):
    sub_scores[index].append([scores[i], len(sub_scores[index])])
```

**동작 설명**:

1. **GPU → CPU**:
   ```python
   scores.tolist()
   ```
   - Torch 텐서 → Python 리스트

2. **Impression별 그룹화**:
   ```python
   sub_scores = [[] for _ in range(indices[-1] + 1)]
   ```
   - 각 impression마다 빈 리스트 생성
   - `indices[-1] + 1`: impression 개수

3. **점수와 원래 순서 저장**:
   ```python
   sub_scores[index].append([scores[i], len(sub_scores[index])])
   ```
   - `scores[i]`: 해당 뉴스의 점수
   - `len(sub_scores[index])`: 해당 impression 내 순서
   - **예시**:
     ```python
     impression 0: [[0.8, 0], [0.6, 1], [0.9, 2]]
     # 0번째 뉴스: 점수 0.8, 원래 순서 0
     # 1번째 뉴스: 점수 0.6, 원래 순서 1
     # 2번째 뉴스: 점수 0.9, 원래 순서 2
     ```

### 6.4 랭킹 계산

**파일**: `util.py:56-62`

```python
with open(result_file, 'w', encoding='utf-8') as result_f:
    for i, sub_score in enumerate(sub_scores):
        sub_score.sort(key=lambda x: x[0], reverse=True)
        result = [0 for _ in range(len(sub_score))]
        for j in range(len(sub_score)):
            result[sub_score[j][1]] = j + 1
        result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
```

**랭킹 생성 과정**:

1. **점수 기준 정렬**:
   ```python
   sub_score.sort(key=lambda x: x[0], reverse=True)
   ```
   - 점수 내림차순 정렬
   - **예시**: `[[0.9, 2], [0.8, 0], [0.6, 1]]`

2. **원래 순서로 랭킹 배치**:
   ```python
   for j in range(len(sub_score)):
       result[sub_score[j][1]] = j + 1
   ```
   - `sub_score[j][1]`: 원래 순서
   - `j + 1`: 랭킹 (1-based)
   - **예시**:
     ```python
     정렬 후: [[0.9, 2], [0.8, 0], [0.6, 1]]
     result[2] = 1  # 2번째 뉴스가 1등
     result[0] = 2  # 0번째 뉴스가 2등
     result[1] = 3  # 1번째 뉴스가 3등
     # result = [2, 3, 1]
     ```

3. **파일 쓰기**:
   ```python
   result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
   ```
   - **형식**: `impression_id [rank1,rank2,rank3,...]`
   - **예시**: `1 [2,3,1]`
   - `.replace(' ', '')`: 리스트 내 공백 제거

### 6.5 평가 지표 계산

**파일**: `util.py:63-68`

```python
if dataset != 'large' or mode != 'test':
    with open(mode + '/ref/truth-%s.txt' % dataset, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
        auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)
    return auc, mrr, ndcg5, ndcg10
else:
    return None, None, None, None
```

**Ground Truth 파일**:
```
1 [1,0,0,1,0]
2 [0,1,0,0]
3 [1,0,1,0,1]
```
- 1: 클릭한 뉴스
- 0: 클릭하지 않은 뉴스

**평가 지표**:

1. **AUC (Area Under the ROC Curve)**:
   - ROC 곡선 아래 면적
   - 0.5 (랜덤) ~ 1.0 (완벽)
   - **의미**: 랜덤한 positive가 랜덤한 negative보다 높은 순위를 받을 확률

2. **MRR (Mean Reciprocal Rank)**:
   - **수식**: `MRR = (1/|Q|) * Σ(1/rank_i)`
   - `rank_i`: 첫 번째 relevant item의 순위
   - **예시**:
     - Positive가 3등 → 1/3 = 0.333
     - Positive가 1등 → 1/1 = 1.0

3. **nDCG@5 (Normalized Discounted Cumulative Gain at 5)**:
   - 상위 5개 추천의 품질 측정
   - **수식**:
     ```
     DCG@k = Σ(rel_i / log2(i + 1))
     nDCG@k = DCG@k / IDCG@k
     ```
   - `rel_i`: i번째 아이템의 relevance (0 or 1)
   - `IDCG`: Ideal DCG (최적 순서일 때)
   - **의미**: 상위 랭킹에 positive가 많을수록 높은 점수

4. **nDCG@10**: nDCG@5와 동일, 상위 10개

**왜 여러 지표?**:
- 각 지표마다 다른 측면 평가
- AUC: 전체 순위 품질
- MRR: 첫 번째 relevant 빠르게 찾기
- nDCG: 상위 k개의 품질

---

## 7. 모델 저장 및 체크포인트 관리

### 7.1 체크포인트 저장

**파일**: `trainer.py:182-183`

```python
if self.epoch_not_increase == 0:
    torch.save({model.model_name: model.state_dict()}, self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch))
```

**저장 형식**:
```python
{
    'NAML-ATT': {
        'word_embedding.weight': tensor(...),
        'title_conv.weight': tensor(...),
        ...
    }
}
```
- 딕셔너리 형태
- Key: 모델 이름
- Value: state_dict

**파일 경로**:
```
models/small/NAML-ATT/#1/NAML-ATT-5
```
- `models`: 루트 디렉토리
- `small`: 데이터셋 크기
- `NAML-ATT`: 모델 이름
- `#1`: run index
- `NAML-ATT-5`: epoch 5 모델

### 7.2 Best Model 관리

**파일**: `trainer.py:191`

```python
shutil.copy(self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch), self.best_model_dir + '/' + model.model_name)
```

**동작**:
1. 훈련 완료 후
2. 최고 성능 epoch의 모델 복사
3. Best model 디렉토리에 저장

**파일 경로**:
```
best_model/small/NAML-ATT/#1/NAML-ATT
```
- Epoch 번호 없음
- 해당 run의 최종 모델

### 7.3 모델 로드

**파일**: `main.py:52-54`

```python
def test(config: Config, mind_corpus: MIND_Corpus):
    model = Model(config)
    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
```

**로드 과정**:

1. **모델 구조 생성**:
   ```python
   model = Model(config)
   ```
   - 빈 파라미터로 모델 인스턴스 생성
   - 구조만 정의됨

2. **체크포인트 로드**:
   ```python
   torch.load(config.test_model_path, map_location=torch.device('cpu'))
   ```
   - `map_location='cpu'`: GPU → CPU 매핑
   - GPU 없어도 로드 가능

3. **State Dict 적용**:
   ```python
   model.load_state_dict(...)
   ```
   - 저장된 파라미터를 모델에 복사
   - 구조가 일치해야 함

4. **GPU 이동**:
   ```python
   model.cuda()
   ```
   - CPU 모델 → GPU 모델

**map_location 옵션**:
```python
# GPU 0에서 저장, GPU 1에서 로드
torch.load(path, map_location='cuda:1')

# GPU에서 저장, CPU에서 로드
torch.load(path, map_location='cpu')

# 자동 매핑
torch.load(path, map_location=lambda storage, loc: storage)
```

### 7.4 결과 로깅

**파일**: `trainer.py:187-190`

```python
with open('%s/%s-%s-dev_log.txt' % (self.dev_res_dir, model.model_name, self._dataset), 'w', encoding='utf-8') as f:
    f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
    for i in range(len(self.auc_results)):
        f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, self.auc_results[i], self.mrr_results[i], self.ndcg5_results[i], self.ndcg10_results[i]))
```

**로그 파일 형식**:
```
Epoch	AUC	MRR	nDCG@5	nDCG@10
1	0.6234	0.2891	0.3045	0.3678
2	0.6456	0.3012	0.3234	0.3890
3	0.6589	0.3145	0.3401	0.4023
...
```

**용도**:
- 훈련 과정 추적
- 그래프 그리기
- 하이퍼파라미터 튜닝 분석

---

## 8. 전체 실행 흐름

### 8.1 메인 함수

**파일**: `main.py:78-88`

```python
if __name__ == '__main__':
    config = Config()
    mind_corpus = MIND_Corpus(config)
    if config.mode == 'train':
        train(config, mind_corpus)
        config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + config.news_encoder + '-' + config.user_encoder
        test(config, mind_corpus)
    elif config.mode == 'dev':
        dev(config, mind_corpus)
    elif config.mode == 'test':
        test(config, mind_corpus)
```

**실행 모드**:

1. **Train 모드**:
   ```
   Config 생성 → Corpus 로딩 → 훈련 → 테스트
   ```

2. **Dev 모드**:
   ```
   Config 생성 → Corpus 로딩 → 검증
   ```

3. **Test 모드**:
   ```
   Config 생성 → Corpus 로딩 → 테스트
   ```

### 8.2 전체 파이프라인

```
1. 시작
   python main.py --news_encoder NAML --user_encoder ATT
   ↓
2. Config 초기화
   - 하이퍼파라미터 파싱
   - 디렉토리 생성
   - CUDA 설정
   ↓
3. Corpus 전처리
   - 사전 구축
   - 워드 임베딩 로딩
   - 엔티티 임베딩 로딩
   - 그래프 생성
   ↓
4. 모델 초기화
   - News Encoder (NAML) 생성
   - User Encoder (ATT) 생성
   - Click Predictor 생성
   - 파라미터 초기화
   ↓
5. 훈련 루프 (각 Epoch)
   ├─ 5.1 Negative Sampling
   ├─ 5.2 배치별 훈련
   │   ├─ 데이터 로딩
   │   ├─ GPU 전송
   │   ├─ Forward Pass
   │   │   ├─ News Encoding
   │   │   ├─ User Encoding
   │   │   └─ Click Prediction
   │   ├─ Loss 계산
   │   ├─ Backward Pass
   │   ├─ Gradient Clipping
   │   └─ Parameter Update
   ├─ 5.3 검증
   │   ├─ 모델 평가 모드
   │   ├─ 점수 계산
   │   ├─ 랭킹 생성
   │   └─ 지표 계산 (AUC, MRR, nDCG)
   ├─ 5.4 Early Stopping 체크
   └─ 5.5 모델 저장
   ↓
6. 최종 테스트
   ├─ Best Model 로딩
   ├─ 테스트 데이터 평가
   └─ 결과 저장
   ↓
7. 종료
```

### 8.3 데이터 흐름

#### 훈련 데이터 흐름
```
behaviors_raw.tsv + news_raw.tsv
  ↓ [MIND_Corpus 전처리]
train_behaviors: [
  [user_ID, history_news_IDs, history_mask, positive_news_ID, negative_news_IDs, behavior_index],
  ...
]
  ↓ [MIND_Train_Dataset]
negative sampling 후 samples: [
  [positive_ID, neg_ID_1, neg_ID_2, neg_ID_3, neg_ID_4],
  ...
]
  ↓ [DataLoader]
배치 텐서: {
  user_ID: [batch_size],
  user_title_text: [batch_size, max_history_num, max_title_length],
  news_title_text: [batch_size, 1+neg_num, max_title_length],
  ...
}
  ↓ [Model Forward]
logits: [batch_size, 1+neg_num]
  ↓ [Loss Function]
loss: scalar
  ↓ [Optimizer]
파라미터 업데이트
```

#### 평가 데이터 흐름
```
behaviors_raw.tsv + news_raw.tsv
  ↓ [MIND_Corpus 전처리]
dev_behaviors: [
  [user_ID, history_news_IDs, history_mask, candidate_news_ID, behavior_index],
  ...
]
  ↓ [MIND_DevTest_Dataset]
각 candidate마다 샘플: [
  [user_ID, history, candidate_1],
  [user_ID, history, candidate_2],
  [user_ID, history, candidate_3],
  ...
]
  ↓ [DataLoader]
배치 텐서: {
  user_ID: [batch_size],
  user_title_text: [batch_size, max_history_num, max_title_length],
  news_title_text: [batch_size, 1, max_title_length],
  ...
}
  ↓ [Model Forward (no_grad)]
scores: [batch_size, 1]
  ↓ [점수 그룹화]
impression별 scores: [
  impression_0: [score_1, score_2, ...],
  impression_1: [score_1, score_2, score_3, ...],
  ...
]
  ↓ [랭킹 계산]
rankings: [
  impression_0: [rank_1, rank_2, ...],
  impression_1: [rank_1, rank_2, rank_3, ...],
  ...
]
  ↓ [평가 지표 계산]
AUC, MRR, nDCG@5, nDCG@10
```

### 8.4 메모리 관리

**GPU 메모리 최적화**:

1. **그래디언트 축적 방지**:
   ```python
   optimizer.zero_grad()
   ```

2. **캐시 비우기**:
   ```python
   torch.cuda.empty_cache()
   ```

3. **평가 시 autograd 비활성화**:
   ```python
   with torch.no_grad():
       ...
   ```

4. **불필요한 텐서 제거**:
   ```python
   del model
   gc.collect()
   ```

### 8.5 분산 훈련 (옵션)

**파일**: `trainer.py:209-389`

```python
def distributed_train(rank, model: nn.Module, config: Config, mind_corpus: MIND_Corpus, run_index: int):
    world_size = config.world_size
    model_name = model.model_name
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    config.device_id = rank
    config.set_cuda()
    model.cuda()
    model = DDP(model, device_ids=[rank])
```

**분산 훈련 개념**:

1. **DDP (DistributedDataParallel)**:
   - 각 GPU에 모델 복사본
   - 각 GPU가 다른 배치 처리
   - 그래디언트 동기화
   - 파라미터 업데이트

2. **NCCL Backend**:
   - NVIDIA Collective Communications Library
   - GPU 간 통신 최적화

3. **Process Group**:
   - `world_size`: 총 프로세스 수 (GPU 수)
   - `rank`: 현재 프로세스 ID (0부터 시작)

4. **데이터 분할**:
   ```python
   batch_size = config.batch_size // world_size
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
   ```
   - 각 GPU가 전체 배치의 일부만 처리
   - 중복 없이 데이터 분할

---

## 부록 A: 핵심 PyTorch 개념

### A.1 텐서 (Tensor)

**기본 생성**:
```python
# 영 텐서
torch.zeros([3, 4])

# 랜덤 텐서
torch.randn([2, 3])  # 정규분포

# 리스트에서 생성
torch.tensor([1, 2, 3])

# NumPy 배열에서
torch.from_numpy(np_array)
```

**텐서 연산**:
```python
# 요소별 연산
a + b, a * b, a / b

# 행렬 곱셈
torch.matmul(a, b)
torch.bmm(a, b)  # 배치 행렬 곱셈

# 차원 조작
a.view([2, 3])
a.reshape([2, 3])
a.permute([1, 0, 2])
a.unsqueeze(dim=1)
a.squeeze(dim=1)
```

**텐서 속성**:
```python
a.shape  # 형태
a.dtype  # 데이터 타입
a.device  # CPU/GPU
a.requires_grad  # 그래디언트 계산 여부
```

### A.2 Autograd (자동 미분)

**기본 사용**:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7.0
```

**Computational Graph**:
```
x (requires_grad=True)
  ↓ (** 2)
x^2
  ↓ (+ 3*x)
x^2 + 3*x
  ↓ (+ 1)
y
  ↓ backward()
그래디언트 계산
```

**다중 변수**:
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2.0, 4.0]
```

### A.3 nn.Module

**커스텀 모듈**:
```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**주요 메서드**:
- `__init__()`: 레이어 초기화
- `forward()`: Forward pass 정의
- `.train()`: 훈련 모드
- `.eval()`: 평가 모드
- `.parameters()`: 모든 파라미터 반환
- `.cuda()`: GPU로 이동

### A.4 DataLoader

**기본 사용**:
```python
dataset = MyDataset()
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for batch in dataloader:
    # 배치 처리
    pass
```

**병렬 로딩**:
- `num_workers > 0`: 멀티프로세싱
- 각 워커가 독립적으로 데이터 로딩
- 메인 프로세스는 GPU 연산에 집중

### A.5 Optimizer

**주요 옵티마이저**:
```python
# SGD (Stochastic Gradient Descent)
optim.SGD(params, lr=0.01, momentum=0.9)

# Adam
optim.Adam(params, lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with decoupled weight decay)
optim.AdamW(params, lr=0.001, weight_decay=0.01)
```

**학습률 스케줄러**:
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # 학습률 업데이트
```

---

## 부록 B: 디버깅 팁

### B.1 텐서 형태 확인

```python
print(f"Tensor shape: {tensor.shape}")
print(f"Min: {tensor.min()}, Max: {tensor.max()}")
print(f"Mean: {tensor.mean()}, Std: {tensor.std()}")
```

### B.2 그래디언트 체크

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
    else:
        print(f"{name}: no gradient")
```

### B.3 NaN 감지

```python
if torch.isnan(loss):
    print("NaN detected in loss!")
    # 디버깅 코드
```

### B.4 GPU 메모리 추적

```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## 부록 C: 성능 최적화

### C.1 혼합 정밀도 훈련 (Mixed Precision)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(input)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**장점**:
- FP16 사용 → 메모리 절약
- 더 큰 배치 가능
- 훈련 속도 향상

### C.2 그래디언트 축적 (Gradient Accumulation)

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    output = model(input)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**효과**:
- 작은 배치 여러 번 → 큰 배치 효과
- GPU 메모리 부족 시 유용

### C.3 DataLoader 최적화

```python
DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
```

**파라미터 설명**:
- `prefetch_factor`: 미리 로딩할 배치 수
- `persistent_workers`: 워커 프로세스 유지

---

## 결론

이 문서는 NAML 모델의 훈련, 평가, 저장 전 과정을 상세히 설명했습니다:

1. **데이터 전처리**: 원본 데이터 → 텐서 변환
2. **모델 아키텍처**: News Encoder + User Encoder + Click Predictor
3. **훈련 과정**: Forward → Loss → Backward → Update
4. **평가 과정**: 점수 계산 → 랭킹 → 지표 산출
5. **모델 관리**: 체크포인트 저장 및 로딩

**핵심 개념**:
- PyTorch의 `nn.Module`, `DataLoader`, `Optimizer`
- Autograd와 Backpropagation
- Attention Mechanism과 Multi-view Learning
- Early Stopping과 Model Checkpointing

**다음 단계**:
- 하이퍼파라미터 튜닝
- 다른 News/User Encoder 실험
- 앙상블 기법 적용
- 실시간 추천 시스템 구축
