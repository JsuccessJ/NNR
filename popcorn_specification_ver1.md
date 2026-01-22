# POPCORN: POPularity-aware COntrastive Interest Matching for News Recommendation

## 논문 제목
**Is This Popular News a Bias?: Popularity-aware Contrastive Interest Matching for News Recommendation**

## 목차
- [1. 연구 동기 및 배경](#1-연구-동기-및-배경)
- [2. 문제 정의](#2-문제-정의)
- [3. Challenges](#3-challenges)
- [4. Preliminary Analysis](#4-preliminary-analysis)
- [5. 제안 모델: POPCORN](#5-제안-모델-popcorn)
- [6. 기존 연구 비교](#6-기존-연구-비교)
- [7. 참고문헌](#7-참고문헌)

---

## 1. 연구 동기 및 배경

### 1.1 뉴스 기사의 정보 구성

뉴스 기사는 크게 다음과 같은 정보를 가지고 있음:
- **Contents 정보**: title, body, category 등
- **Context 정보**: popularity, freshness, dwell-time 등
  - Context: 시간이 지남에 따라 달라지는 뉴스기사의 dynamic information [1]

**Content Feature vs Context Feature:**

| Content Feature | Context Feature |
|----------------|-----------------|
| Semantic | Popularity |
| Topic Model | CTR |
| Entity | Recency |
| Keyword | Novelty |
| Emotion | Dwell Time |
| Multimodal | Bias |

- Content Feature: 뉴스 컨텐츠로부터 추출
- Context Feature: Dynamic information

### 1.2 Popularity Bias in News Recommendation

**문제 상황:**
- 대부분의 뉴스추천 방법은 유저의 클릭 히스토리를 기반으로 선호도를 표현하고, 향후에 클릭할 뉴스를 예측함
- Real-world에서 클릭 히스토리에는 유저의 관심사와 관계없는 인기 있는 뉴스가 포함될 수 있음 [2-5]
- 이로 인해 뉴스추천 시스템은 인기 뉴스도 그 유저의 관심사라고 잘못 이해하게 되고, 유저의 관심사가 아닌 인기 뉴스를 더 추천하게 되는 **인기 편향(popularity bias)**이 발생함

**예시:**
- 친구들이 연예인 스캔들 얘기를 많이하자 유저 A도 뉴스 플랫폼에서 관련 주제의 뉴스를 클릭
- 하지만 본래 관심사는 경제 뉴스였음
- 인기 때문에 클릭이 발생
- 클릭 로그에 bias가 생김
- 이후 연예인 스캔들 주제의 뉴스는 유저 A에게 더 많이 추천됨

**중요성:**
- 따라서 유저의 관심사와 잘 매칭되는 뉴스를 제공하기 위해 인기 편향을 완화하는 것은 뉴스추천에 매우 중요함

---

## 2. 문제 정의

### 2.1 연구 목표
**Goal:** 유저의 클릭 히스토리를 기반으로, 그 유저의 선호도를 정확하게 표현하는 것

### 2.2 기존 연구의 한계
기존 뉴스추천 방법들이 높은 정확도를 달성했지만, 다음과 같은 중요한 challenge들은 여전히 탐구되지 않음

---

## 3. Challenges

### Challenge 1 (C1): Hot News의 구분 필요 (Differentiation of hot news)

#### 3.1.1 기존 연구의 한계
- 기존 연구들은 유저의 선호도를 표현할 때 유저 히스토리에 있는 clicked hot news를:
  1. 편향으로 **전혀 간주하지 않거나**
  2. **모두 편향으로 간주함** [1-8]
- **Hot news 정의:** 유저들에게 많은 클릭을 받은 popular한 뉴스기사

#### 3.1.2 본 연구의 관점: Hot News의 2가지 Case

Looking more closely, 유저 히스토리에 있는 hot news는 다음과 같이 2가지 case로 구분할 수 있음:

**Case 1: Hot news as Interest Signal**
- 유저의 관심사를 나타내는, interest signal을 드러내는 hot news
- 예시:
  - 본래 선호하는 컨텐츠인데 우연히 hot한 경우
  - 처음엔 hot해서 클릭했지만 이후 선호하는 컨텐츠로 발전한 경우

**Case 2: Hot news as Popularity Bias**
- 유저의 관심사가 아닌, popularity bias로 작용하는 hot news
- 예시:
  - 인기로 인해 클릭이 유도된 경우

#### 3.1.3 결론
따라서, 유저의 선호도를 정확하게 표현하기 위해서는 유저가 클릭한 hot news를 정교하게 이해해서 **hot news들 중 interest signal을 가지는 뉴스와 popularity bias를 가지는 뉴스를 carefully 구분해야 함**

---

### Challenge 2 (C2): Hot User의 이해 필요 (Comprehension of hot users)

#### 3.2.1 기존 연구의 분류
- 기존 연구들은 일반적으로 유저를 warm user와 cold user로 분류함 [6-8]
- Warm user는 cold user보다 그 유저의 선호도를 더 잘 표현할 수 있음

#### 3.2.2 본 연구의 관점: Hot User의 발견

Looking more closely, 우리는 warm user들 중에서도 클릭 수가 지나치게 많은 user들이 오히려 선호도 추론을 더 어렵게 만든다는 것을 관찰함 (See 사전실험 P2). 그게 바로 **hot user**들임

**Hot User 정의:**
- 많은 수의 뉴스기사를 클릭한 유저 (e.g., 상위 K(e.g., 5)% CTR 이상인 유저)
- 예시:
  - Cold user: 뉴스기사를 1-4개 클릭
  - Warm user: 5-19개 클릭
  - Hot user: 20개 이상 클릭

**유저 타입별 특성:**
- Cold User: User with **only a few** news
- Warm User: User with **many** news
- Hot User: User with **too many** news

#### 3.2.3 왜 Hot User는 선호도 추론을 어렵게 만드는가?

**Having diverse interests:**
- Hot user는 다양한 토픽에 분산된 관심을 보임
- 데이터 분석 결과:
  - **Topics:** Warm 2.1개 vs. Hot 4.2개
  - **Sub-topics:** Warm 4.8개 vs. Hot 16.5개
- 이러한 다양하고 분산된 관심사는 hot user의 선호도 추론을 보다 어렵게 함

**정확도 저하:**
- 이러한 이슈들로 인해, hot user들은 warm user들보다 뉴스추천 정확도를 저하시킴
- See P2: Hot user의 평균 정확도 오류율이 warm user보다 **2.8배** 더 높다는 것을 확인

#### 3.2.4 결론
따라서, 유저의 선호도를 정확하게 표현하기 위해서는 **선호도 추론을 어렵게 하는 hot user들을 유저 모델링에 carefully 반영해야 함**

---

### Challenge 3 (C3): 토픽별 상대적인 인기도 고려 필요 (Consideration of relative popularity across topics)

#### 3.3.1 기존 연구의 한계
- 기존 연구들의 popularity는 모든 뉴스기사에 대해 동일한 metric (i.e., CTR)으로 측정함 [2-4]

#### 3.3.2 본 연구의 관점: 토픽별 Attractiveness 차이

Looking more closely, 뉴스의 토픽에 따라 유저를 얼마나 끌어들이는지의 정도(attractiveness)가 다를 수 있음 [9-12]

**High-attractiveness topics:**
- 스포츠, 연예, 자연재해와 같은 토픽의 뉴스기사
- 여행, 건강 뉴스보다 상대적으로 더 많은 유저들에게 클릭을 받음
- 이유: 상대적으로 더 자극적이고 시의적인 내용을 포함하기 때문 [9, 10]
  - 이로 인해 짧은 시간에 많은 클릭이 유도됨

**Low-attractiveness topics:**
- 여행, 건강, 금융 뉴스
- 상대적으로 더 유용한, 깊이 있는 정보를 포함해서 더 적은 클릭을 받음 [11, 12]

#### 3.3.3 문제점

이러한 차이를 고려하지 않은 고정된 인기도는 real-world의 상대적인 인기도를 제대로 반영하지 못함

**예시:**
- 건강 뉴스의 500 clicks: 해당 토픽 내에서 매우 인기 (건강 뉴스 내 상위 5%)
- 자연재해 뉴스의 500 clicks: 해당 토픽 내에서 보통 (자연재해 뉴스 내 중위 50%)

#### 3.3.4 결론
따라서 뉴스추천에서 real-world에 더 align한 인기도를 반영하기 위해서는 **뉴스기사의 popularity를 토픽에 따라 정의하고, 이를 통해 상대적인 인기도를 고려해야 함**

이는 (C1) hot news의 구분과 (C2) hot user의 이해를 더욱 정교하게 하는 데에 도움을 줌

---

## 4. Preliminary Analysis

우리는 Motivation 섹션에서 앞서 설명한 (C1) hot news를 구분하고, (C2) hot user를 이해하는 것이 실세계에서 뉴스추천에 실제로 도움이 될 수 있는지 검증하고, (C3) 토픽별 상대적인 인기도를 확인하고자 함

### 4.1 (P1) for (C1) Differentiation of hot news

#### 4.1.1 목표
Hot news가 interest signal과 popularity bias로 구분될 수 있는지 확인하고, 이 구분이 기존 뉴스추천 정확도 개선에 기여할 수 있는지 검증

#### 4.1.2 Hot news 정의 (사전실험용)
- **정의:** 클릭시점 기준 지난 12시간 내 CTR(클릭수/노출수) 상위 5%인 뉴스
- **K=5 결정 근거:** CTR 분포 분석 결과, 상위 5% 이후로 평균 CTR이 급격히 증가하는 것을 관찰
- **Note:** 이는 사전실험을 위한 정의이며, 실제 모델링에서는 popularity를 연속값으로 사용

#### 4.1.3 분석 방법
- 기존 방법(e.g., CROWN)의 추천 결과로부터 clicked hot news를 분석 (in Adressa test set)
- 유저 히스토리에 있는 hot news가 interest signal을 드러내는지, popularity bias로 작용하는지 판단 기준:
  1. 추천 정확도를 올리는 데 기여하면 → **interest signal**
  2. 추천 정확도를 떨어뜨리는 데 기여하면 → **popularity bias**

#### 4.1.4 분석 결과

**A. True Positive 샘플 분석: Hot news의 두 가지 영향(interest/bias)을 보기 위함**
- 모든 정답 샘플의 clicked hot news 분석 결과:
  - 약 **42%의 hot news**는 TP 후보뉴스와 높은 컨텐츠 유사도 (일반 클릭 뉴스보다 높음)
    - → 추천 정확도를 올리는 데 기여 → **interest signal**
  - 약 **58%의 hot news**는 TP 후보뉴스와 낮은 컨텐츠 유사도 (일반 클릭 뉴스보다 낮음)
    - → 추천 정확도를 떨어뜨리는 데 기여 → **popularity bias**

**B. False Positive 샘플 분석: Bias의 영향을 보기 위함**
- 약 **63%의 hot news**가 FP 유도에 기여
- FP 후보뉴스와 높은 유사도 (인기로 인한 오판)

**C. False Negative 샘플 분석: 놓친 Interest를 보기 위함**
- 약 **45%의 hot news**가 FN 후보뉴스와 높은 유사도
- "강조되었다면 정답을 맞출 수 있었던 interest signal"

#### 4.1.5 인사이트
Hot news는 interest signal과 popularity bias로 구분되며, **이들을 정교하게 구분하는 것이 뉴스추천 정확도 개선에 기여할 수 있음**을 확인함

---

### 4.2 (P2) for (C2) Differentiation of hot users

#### 4.2.1 목표
Hot user들이 warm user들보다 실제로 선호도 추론이 어려운 user인지 확인하고, 이를 address하는 것이 정확도 개선에 기여할 수 있는지 검증

#### 4.2.2 Cold/Warm/Hot user 정의
- **Cold user:** 하위 M(e.g., 20)% 클릭 수 미만인 유저 (20%)
  - 데이터셋마다 활동량 분포가 달라서 고정수보다 분위수가 적절함
- **Warm user:** 하위 M% 클릭 수 이상, 상위 M% 클릭 수 미만인 유저 (60%)
- **Hot user:** 상위 M% 클릭 수 이상인 유저 (20%)

#### 4.2.3 분석 방법
Hot user가 warm user보다 선호도 추론이 어려운 user인지는:
- Hot user가 추천 정확도를 떨어뜨리는 데 더 기여하는지 확인해보면 알 수 있음

#### 4.2.4 분석 결과

**A. Warm user와 Hot user 각 비율에서 오류 비율을 확인 (baseline: CROWN)**
- **Adressa:**
  - Warm user: 8.2/60% (13.6%)
  - Hot user: 7.6/20% (37.8%)
  - → **2.8배** 차이
- **MIND:**
  - Warm user: 13.8/60% (23.1%)
  - Hot user: 10.3/20% (51.7%)
  - → **2.2배** 차이
- **결론:** Hot user가 warm user보다 오류 비율이 높다는 건, real-world에서 실제로 선호도 추론이 어려운 유저들임을 나타냄
  - 이를 address하는 것은 뉴스추천 정확도 개선에 직접적인 도움이 될 수 있음

**B. 왜 Hot user가 선호도 추론을 어렵게 만드는지에 대한 Evidence**

**(1) Accumulation of hot news: Warm과 Hot user의 인기뉴스 클릭 수 비교**
- Warm/Hot 유저의 클릭 히스토리에 있는 hot news(top 5%) 개수의 평균을 계산
- **Adressa:** Warm 4.7개 → Hot 20.3개 (**4.28배** 증가)
- **MIND:** Warm 11.4개 → Hot 37.5개 (**3.28배** 증가)

**(2) Dispersion of diverse interests: Warm과 Hot user의 토픽 다양성 비교**
- Warm/Hot user의 클릭 히스토리에 있는 unique한 토픽 개수의 평균을 계산
- **Adressa:** Warm 2.1개 → Hot 4.2개 (**2.03배** 증가)
- **MIND:** Warm 6.0개 → Hot 9.7개 (**1.60배** 증가)

#### 4.2.5 인사이트 정리
- Hot user는 warm user보다 선호도 추론이 어려운 유형임을 실험적으로 확인함
- **Hot user를 구분하고, 이들의 특성을 정교하게 반영한 모델링 전략이 필요함**을 확인함

**Hot User Analysis 시각화:**
- **Error Ratio by User Group:** Hot user의 오류 비율이 Warm user보다 현저히 높음
- **Topics & Hot News in History:** Hot user는 더 많은 토픽과 hot news를 클릭함

---

### 4.3 (P3) for (C3) Consideration of relative popularity across topics

#### 4.3.1 목표
Real-world 상에서 실제로 토픽별 인기도 차이가 있는지 확인

#### 4.3.2 분석: 토픽별 평균 CTR (i.e., 인기도) 비교 (Adressa/MIND)

**High-attractiveness topics:**
- Sports, Entertainment, Disaster (평균 CTR: XX) - **TODO: 실험 데이터 입력**

**Low-attractiveness topics:**
- Health, Travel, Finance (평균 CTR: XX) - **TODO: 실험 데이터 입력**

#### 4.3.3 동일한 CTR(클릭수/노출수)의 의미
- Health 뉴스 0.3 → 상위 5% (매우 인기)
- Sports 뉴스 0.3 → 중위 50% (보통)

#### 4.3.4 인사이트 정리
- 실세계에서 토픽에 따라 인기도에 차이가 있음을 발견
- **토픽별 상대적인 인기도 차이를 고려하지 않으면, 동일한 인기도가 토픽에 따라 의미가 달라질 수 있음**

**Visualization Example:** Topic-wise differences in news lifetime
- 토픽별 lifetime(hours) 차이를 보여주는 그래프 참조

---

### 4.4 추가 분석: 실세계 데이터셋의 유저 및 클릭 분포

**(Appendix 추가 예정) 실세계 데이터셋(e.g., Adressa, MIND)의 유저 및 클릭 분포 분석**

#### 4.4.1 유저 그룹 분포
- **Adressa:**
  - Cold 34.4% (0 클릭)
  - Warm 45.5% (평균 8.4 클릭)
  - Hot 20.1% (평균 147 클릭)
- **MIND:**
  - Cold 21.5% (평균 3.8 클릭)
  - Warm 58.5% (평균 25 클릭)
  - Hot 20.1% (평균 318 클릭)

#### 4.4.2 누적 클릭 분포
- **Adressa:** 클릭 수 상위 20% 유저(i.e., hot user)가 전체 클릭의 **88.5%** 차지
- **MIND:** 클릭 수 상위 20% 유저(i.e., hot user)가 전체 클릭의 **80.4%** 차지
- **Heavy-tailed 분포**로 hot user에게 클릭이 집중됨 → 뉴스추천에 큰 영향을 주는 유형

#### 4.4.3 추가 인사이트
**Hot user에게 대부분의 클릭(약 80%)이 집중되어 뉴스추천에 큰 영향을 주는 유형임을 확인함**

**Cumulative Click Distribution 시각화:**
- Adressa (Top 20%, 88.5% clicks)
- MIND (Top 20%, 80.4% clicks)

---

## 5. 제안 모델: POPCORN

### 5.1 Overview

Motivated by these challenges, 우리는 개인화 뉴스추천을 위한 novel framework인 **POPCORN (POPularity-aware COntrastive Interest Matching for News Recommendation)**을 제안함

**핵심 특징:**
- 제안하는 framework는 뉴스인코더와 유저인코더에 대해 **model-agnostic**함
- 크게 **4가지 컴포넌트**로 구성됨:
  - **(I1)** Popularity-disentangled News Modeling
  - **(I2)** Hot-to-Warm User Modeling
  - **(I3)** Topic-wise Popularity Modeling
  - **(I4)** Popularity-aware Contrastive Interest Matching

---

### 5.2 Component (I1): Popularity-disentangled News Modeling

**목적:** Challenge 1 (C1) 해결 - Hot news를 interest signal과 popularity bias로 구분

#### Module 1: News Encoder

**설계:**
- **Plug-in design:** 기존 방법들 중 뉴스인코더가 있는 어떤 방법도 갈아끼워서 사용 가능
- **Input:** 뉴스 타이틀 (title) - 우선적으로 title 사용
  - 추후 확장 시 abstract, body 등 추가 가능
- **Output:** h_j ∈ ℝ^d (뉴스 임베딩)

**구현 상세:**
- PLM 기반 인코더 (BERT, RoBERTa 등) 또는 기존 뉴스추천 모델의 인코더 사용
- Title tokenization → Embedding → Encoder → Pooling → h_j

---

#### Module 2: Popularity Disentangler

**목표:**
Hot news를 (interest의 clue가 되는) popularity-free한 컨텐츠 정보와 (인기도를 결정하는, i.e., popularity bias의 clue가 되는) 컨텐츠 정보로 분리

**아키텍처:**

```
News Encoder → h_j ∈ ℝ^d
    ├─→ Popularity-free Decoder → f_j ∈ ℝ^d (pop-free news rep.) → Popularity Predictor → ỹ_jf
    └─→ Popularity-aware Decoder → p_j ∈ ℝ^d (pop-aware news rep.) → Popularity Predictor → ŷ_jp
```

**Popularity-free Decoder:**

**목적:** 인기도와 무관한 순수 콘텐츠 특성만 추출

**구조:**
```
d^1_j = LeakyReLU(Dense_f1(h_j))           # 첫 번째 projection layer
f_j = LeakyReLU(Dense_f2([d^1_j ; h_j]))   # residual connection으로 정보 보존
```

**어떻게 f_j가 도출되는가?**
1. **h_j ∈ ℝ^d를 input으로** 첫 번째 Dense layer 통과
   - Dense_f1: ℝ^d → ℝ^d
   - LeakyReLU 활성화 함수 적용
   - → **d^1_j ∈ ℝ^d** 중간 표현 생성

2. **[d^1_j ; h_j] concatenation**으로 원본 정보 보존 (residual connection)
   - [d^1_j ; h_j] ∈ ℝ^(2d)

3. 두 번째 Dense layer 통과
   - Dense_f2: ℝ^(2d) → ℝ^d
   - LeakyReLU 활성화 함수 적용
   - → **f_j ∈ ℝ^d 최종 출력**

**Popularity-aware Decoder:**

**목적:** 인기도를 결정하는 콘텐츠 특성 추출

**구조:**
```
d^1_j = LeakyReLU(Dense_p1(h_j))           # 첫 번째 projection layer
p_j = LeakyReLU(Dense_p2([d^1_j ; h_j]))   # residual connection으로 정보 보존
```

**어떻게 p_j가 도출되는가?**
1. **h_j ∈ ℝ^d를 input으로** 첫 번째 Dense layer 통과
   - Dense_p1: ℝ^d → ℝ^d
   - LeakyReLU 활성화 함수 적용
   - → **d^1_j ∈ ℝ^d** 중간 표현 생성

2. **[d^1_j ; h_j] concatenation**으로 원본 정보 보존 (residual connection)
   - [d^1_j ; h_j] ∈ ℝ^(2d)

3. 두 번째 Dense layer 통과
   - Dense_p2: ℝ^(2d) → ℝ^d
   - LeakyReLU 활성화 함수 적용
   - → **p_j ∈ ℝ^d 최종 출력**

**핵심:**
- **두 Decoder는 동일한 구조**를 가지지만, **파라미터는 독립적**
- Forward pass만 보면 f_j와 p_j는 단순히 h_j의 두 가지 다른 projection
- **차별화는 학습 단계(Loss)에서 발생** (아래 Module 3 참조)

---

#### Module 3: Popularity Predictor

**목적:** 토픽 정보를 함께 활용하여 인기도 예측 (C3 해결에도 기여)

**Topic Embedding 생성 (requirement #3):**
```
topic_emb_j = Dense([category_emb_j ; subcategory_emb_j])
```
- **category_emb_j**: 뉴스의 주 카테고리 임베딩 (e.g., Sports, Politics, Entertainment)
- **subcategory_emb_j**: 뉴스의 세부 카테고리 임베딩 (e.g., Football, Election, Celebrity)
- **Concatenate** 후 Dense layer로 projection하여 최종 topic_emb_j ∈ ℝ^d_t 생성
- 이를 통해 계층적 토픽 정보를 모두 활용

**Popularity의 정의 및 Classification 설정:**

Popularity는 뉴스의 클릭 수를 최대 클릭 수로 나눈 **연속값 [0, 1]**이지만,
이를 **10개 클래스로 binning하여 classification 문제**로 변환:

```python
# Popularity binning (10 classes)
if 0.0 ≤ popularity < 0.1: class = 0
if 0.1 ≤ popularity < 0.2: class = 1
if 0.2 ≤ popularity < 0.3: class = 2
...
if 0.9 ≤ popularity ≤ 1.0: class = 9
```

**이유:**
- 기존 연구들(PENR 등)에서 검증된 방식
- Regression보다 robust한 학습
- Ordinal classification으로 인기도 구간 학습 가능

**Input:** `[p_j or f_j ; topic_emb_j]` ∈ ℝ^(d + d_t)
**Output:** predicted popularity distribution `ŷ_jp` or `ỹ_jf` ∈ ℝ^10 (10-class probability)

**구현 버전:**

**(Naive ver.) → 우선 이걸로 먼저 구현**
```
ŷ_jp or ŷ_jf = Softmax(Dense([p_j or f_j ; topic_emb_j]))  # shape: (10,)
```

**Popularity Predictor 출력 도출 원리 (requirement #4):**

1. **Input 준비:**
   - news representation: p_j 또는 f_j ∈ ℝ^d
   - topic_emb_j ∈ ℝ^d_t (category + subcategory 정보 포함)
   - Concatenate: [p_j or f_j ; topic_emb_j] ∈ ℝ^(d + d_t)

2. **Feed Forward Network:**
   ```
   h_pred = ReLU(Dense_pred1([p_j or f_j ; topic_emb_j]))    # intermediate representation
   logits = Dense_pred2(h_pred)                              # shape: (10,)
   ŷ_jp or ŷ_jf = Softmax(logits)                           # shape: (10,) probability distribution
   ```

3. **출력값의 의미:**
   - `ŷ_jp[i]` = 뉴스 j가 인기도 구간 i에 속할 확률
   - 예: ŷ_jp = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.05, 0.02]
     → 인기도가 0.5~0.7 구간(class 5, 6)에 있을 확률이 높음

4. **왜 이렇게 동작하는가?**
   - **p_j 사용 시 (L_p 학습)**: p_j가 인기도 관련 특성을 담고 있어 정확한 클래스 예측 가능
   - **f_j 사용 시 (L_a 학습)**: f_j는 인기도 정보가 없어 uniform distribution에 가깝게 예측 실패
     → 이를 통해 f_j에서 인기도 정보 제거
   - **topic_emb_j 포함**: 같은 콘텐츠라도 토픽에 따라 인기도가 다를 수 있음을 반영 (C3 해결)
     - 예: Sports 뉴스는 평균적으로 Entertainment보다 인기도가 높을 수 있음

**(Enhanced ver.) - 추후 확장**
```
g_j = σ(Dense([p_j or f_j ; topic_emb_j]))  # g_j: scalar gate
ŷ_jp or ŷ_jf = Softmax([g_j ⊙ Dense(p_j or f_j) + (1-g_j) ⊙ Dense(topic_emb_j)])
```
- Gate mechanism으로 뉴스 표현과 토픽 정보의 가중치를 동적으로 조절

---

#### 학습 메커니즘: 3 Losses

**(1) Popularity Prediction Loss: L_p**

**목적:** p_j가 (토픽별 상대적) 인기도 정보를 담도록 유도 (prediction error minimize)

```
L_p = -y_j log(ŷ_jp)
```

**상세:**
- `ŷ_jp = Softmax(Dense([p_j ; topic_emb_j]))`
- `y_j` = 실제 인기도 (true popularity label)
- 토픽 임베딩을 함께 입력하여 **C3 해결**

---

**(2) Adversarial Loss: L_a**

**목적:** f_j가 (토픽별 상대적) 인기도 정보를 담지 않도록 유도 (prediction error maximize)

```
L_a = -1 / (y_j log(ỹ_jf))
```

**상세:**
- `ỹ_jf = Softmax(Dense([f_j ; topic_emb_j]))`
- f_j가 인기도를 예측하지 못하도록 adversarial하게 학습

---

**(3) Reconstruction Loss: L_r**

**목적:** 정보 손실 방지

```
L_r = 1/2 * ||Dense([f_j ; p_j]) - h_j||²
```

---

**Total Loss for Popularity Disentangling:**

```
L_pop = (L_r + L_p + L_a)
```

---

#### Loss 함수 구현 설계

**핵심 질문:** f_j와 p_j는 구조가 동일한데 어떻게 다르게 학습되는가?

**답변:**
1. **파라미터 독립성**: decoder_f와 decoder_p는 구조는 같지만 **파라미터는 별도** (W_f1, b_f1, W_f2, b_f2 vs W_p1, b_p1, W_p2, b_p2)
2. **Loss 연결 차별화**: f_j는 L_a, L_r에 연결 / p_j는 L_p, L_r에 연결
3. **Autograd 자동 분리**: PyTorch의 computational graph가 각 파라미터에 해당하는 gradient만 전달

**Gradient Flow:**

| 파라미터 | 연결된 Loss | Gradient 방향 | 학습 목표 |
|---------|------------|--------------|----------|
| **decoder_f** | L_a + L_r | ∂L_a/∂W_f + ∂L_r/∂W_f | • L_a: 인기도 정보 제거 (adversarial)<br>• L_r: 콘텐츠 정보 보존 (복원 기여) |
| **decoder_p** | L_p + L_r | ∂L_p/∂W_p + ∂L_r/∂W_p | • L_p: 인기도 정보 추출 (prediction)<br>• L_r: 인기도 정보 보존 (복원 기여) |

---

**PyTorch 구현 코드:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PopularityDisentanglingLoss(nn.Module):
    """
    목적: f_j와 p_j를 서로 다른 특성(popularity-free vs popularity-aware)으로 분리하는 Loss

    입력:
        - f_j: popularity-free representation (batch_size, d)
        - p_j: popularity-aware representation (batch_size, d)
        - h_j: original news embedding (batch_size, d)
        - logits_p: p_j로부터 예측한 인기도 logits (batch_size, 10)
        - logits_f: f_j로부터 예측한 인기도 logits (batch_size, 10)
        - y_j: 실제 인기도 클래스 label (batch_size,) - 0~9 정수

    출력:
        - L_pop: Total loss for popularity disentangling (scalar)
        - loss_dict: 각 loss 항목의 값 (monitoring용)
    """

    def __init__(self, d):
        super().__init__()
        # Reconstruction을 위한 Dense layer
        self.reconstruct_layer = nn.Linear(2 * d, d)

    def forward(self, f_j, p_j, h_j, logits_p, logits_f, y_j):
        """
        Args:
            f_j: (batch_size, d) - popularity-free representation
            p_j: (batch_size, d) - popularity-aware representation
            h_j: (batch_size, d) - original news embedding
            logits_p: (batch_size, 10) - p_j로부터 예측한 logits (ŷ_jp의 Softmax 이전 값)
            logits_f: (batch_size, 10) - f_j로부터 예측한 logits (ỹ_jf의 Softmax 이전 값)
            y_j: (batch_size,) - 실제 인기도 클래스 (0~9 정수)

        Returns:
            L_pop: scalar - total loss
            loss_dict: dict - {'L_r': float, 'L_p': float, 'L_a': float}
        """

        # (1) Popularity Prediction Loss: L_p
        # 목적: p_j가 인기도를 잘 예측하도록 (prediction error minimize)
        # Input: logits_p (batch_size, 10), y_j (batch_size,)
        # Output: scalar loss
        L_p = F.cross_entropy(logits_p, y_j)
        # → decoder_p의 파라미터에 ∂L_p/∂W_p gradient 전달
        # → p_j가 인기도 정보를 담는 방향으로 학습

        # (2) Adversarial Loss: L_a
        # 목적: f_j가 인기도를 예측하지 못하도록 (prediction error maximize)
        # Input: logits_f (batch_size, 10), y_j (batch_size,)
        # Output: scalar loss (부호 반전)

        # 방법: Negative cross-entropy
        # Step 1: f_j로 예측한 확률 분포 계산
        probs_f = F.softmax(logits_f, dim=-1)  # (batch_size, 10)

        # Step 2: 정답 클래스의 확률 추출
        # probs_f[i, y_j[i]]를 추출 → 정답을 맞출 확률
        pred_correct = probs_f.gather(1, y_j.unsqueeze(1)).squeeze(1)  # (batch_size,)

        # Step 3: 정답을 맞추는 확률을 minimize (= error maximize)
        L_a = -torch.mean(torch.log(pred_correct + 1e-10))
        # 부호: minimize L_a → pred_correct 낮아짐 → f_j가 인기도 예측 못함
        # → decoder_f의 파라미터에 ∂L_a/∂W_f gradient 전달
        # → f_j가 인기도 정보를 제거하는 방향으로 학습

        # (3) Reconstruction Loss: L_r
        # 목적: f_j와 p_j를 합쳐서 h_j를 복원 (정보 손실 방지)
        # Input: f_j (batch_size, d), p_j (batch_size, d), h_j (batch_size, d)
        # Output: scalar loss

        # Step 1: f_j와 p_j concatenate
        f_p_concat = torch.cat([f_j, p_j], dim=-1)  # (batch_size, 2d)

        # Step 2: Dense layer로 h_j 복원
        h_j_recon = self.reconstruct_layer(f_p_concat)  # (batch_size, d)

        # Step 3: MSE loss 계산
        L_r = 0.5 * torch.mean((h_j_recon - h_j) ** 2)
        # → decoder_f와 decoder_p 모두에 ∂L_r/∂W gradient 전달
        # → f_j: 인기도 제외한 콘텐츠 정보 보존
        # → p_j: 인기도 정보 보존

        # Total Loss
        L_pop = L_r + L_p + L_a

        # Monitoring용 dict
        loss_dict = {
            'L_r': L_r.item(),
            'L_p': L_p.item(),
            'L_a': L_a.item()
        }

        return L_pop, loss_dict
```

**사용 예시 (Training Loop):**

```python
# Model components 정의
news_encoder = NewsEncoder(d=256)                     # 뉴스 타이틀 → h_j
disentangler = PopularityDisentangler(d=256)         # h_j → f_j, p_j
popularity_predictor = PopularityPredictor(d=256)    # (f_j or p_j, topic_emb_j) → logits
loss_fn = PopularityDisentanglingLoss(d=256)         # Loss 계산

# Optimizer에 모든 파라미터 등록
optimizer = torch.optim.Adam([
    {'params': news_encoder.parameters()},
    {'params': disentangler.parameters()},           # decoder_f, decoder_p 포함
    {'params': popularity_predictor.parameters()},
    {'params': loss_fn.reconstruct_layer.parameters()}
])

# Training loop
for batch in dataloader:
    # Batch data
    # - news_title_j: (batch_size, seq_len) - 뉴스 타이틀 token IDs
    # - y_j: (batch_size,) - 인기도 클래스 (0~9 정수)
    # - topic_emb_j: (batch_size, d_topic) - 토픽 임베딩
    news_title_j = batch['news_title']
    y_j = batch['popularity_class']
    topic_emb_j = batch['topic_emb']

    # Forward pass
    # Input: news_title_j (batch_size, seq_len)
    # Output: h_j (batch_size, d)
    h_j = news_encoder(news_title_j)

    # Input: h_j (batch_size, d)
    # Output: f_j (batch_size, d), p_j (batch_size, d)
    f_j, p_j = disentangler(h_j)

    # Popularity prediction
    # Input: p_j (batch_size, d), topic_emb_j (batch_size, d_topic)
    # Output: logits_p (batch_size, 10) - ŷ_jp의 Softmax 이전 값
    logits_p = popularity_predictor(p_j, topic_emb_j)

    # Input: f_j (batch_size, d), topic_emb_j (batch_size, d_topic)
    # Output: logits_f (batch_size, 10) - ỹ_jf의 Softmax 이전 값
    logits_f = popularity_predictor(f_j, topic_emb_j)

    # Loss 계산
    # Input: f_j, p_j, h_j, logits_p, logits_f, y_j
    # Output: L_pop (scalar), loss_dict (monitoring용)
    L_pop, loss_dict = loss_fn(f_j, p_j, h_j, logits_p, logits_f, y_j)

    # Backward pass
    optimizer.zero_grad()
    L_pop.backward()
    # PyTorch autograd가 computational graph를 따라:
    # - decoder_f: ∂L_a/∂W_f + ∂L_r/∂W_f 계산하여 .grad에 저장
    # - decoder_p: ∂L_p/∂W_p + ∂L_r/∂W_p 계산하여 .grad에 저장
    optimizer.step()

    # Logging
    print(f"L_pop: {L_pop.item():.4f} | L_r: {loss_dict['L_r']:.4f} | "
          f"L_p: {loss_dict['L_p']:.4f} | L_a: {loss_dict['L_a']:.4f}")
```

**핵심 구현 포인트:**

1. **파라미터 독립성 보장**
   ```python
   class PopularityDisentangler(nn.Module):
       def __init__(self, d):
           self.decoder_f = nn.Sequential(...)  # 독립적 파라미터
           self.decoder_p = nn.Sequential(...)  # 독립적 파라미터
   ```

2. **Loss 연결 차별화**
   - `logits_p = popularity_predictor(p_j, ...)` → L_p에 연결
   - `logits_f = popularity_predictor(f_j, ...)` → L_a에 연결
   - `h_j_recon = reconstruct([f_j ; p_j])` → L_r에 둘 다 연결

3. **Autograd 자동 처리**
   - `L_pop.backward()` 한 번 호출 → 모든 gradient 자동 계산
   - 별도의 backward 분리 불필요

4. **학습 결과**
   - **f_j**: "이 뉴스는 축구 경기 결과다" (popularity-free content)
   - **p_j**: "이 뉴스는 클릭 500회로 토픽 내 인기가 높다" (popularity-aware)

---

**L_pop 적용 시점 및 위치:**

**언제 계산되는가?**
- **매 학습 step마다** News Encoder와 Popularity Disentangler를 통과할 때마다 계산됨
- Training 과정에서 **L_click과 동시에** 계산되어 multi-task learning 수행

**어디에 설계되는가?**
```python
# Training Loop (pseudo-code)
for batch in dataloader:
    # I1: Popularity-disentangled News Modeling
    h_j = NewsEncoder(news_title_j)                  # Browsed news encoding
    h_c = NewsEncoder(news_title_c)                  # Candidate news encoding

    f_j, p_j = PopularityDisentangler(h_j)          # Browsed news disentangling
    f_c, p_c = PopularityDisentangler(h_c)          # Candidate news disentangling

    # Popularity prediction (auxiliary task)
    ŷ_jp = PopularityPredictor(p_j, topic_emb_j)    # p_j로 예측
    ỹ_jf = PopularityPredictor(f_j, topic_emb_j)    # f_j로 예측

    # Calculate L_pop
    L_r = 0.5 * ||Dense([f_j ; p_j]) - h_j||²       # Reconstruction
    L_p = -y_j log(ŷ_jp)                             # p_j가 인기도 예측하도록
    L_a = -1 / (y_j log(ỹ_jf))                       # f_j가 인기도 예측 못하도록
    L_pop = L_r + L_p + L_a

    # I2: Hot-to-Warm User Modeling
    r_j = GatedResidualConnection(f_j, α_j, g_i)    # r_j 생성
    u = UserEncoder({r_1, r_2, ..., r_N})           # User representation

    # I4: Contrastive Interest Matching
    score = ContrastiveMatching(u, f_c, p_c, {p_j}) # CTR 예측
    L_click = -y_click log(score)                    # Click loss

    # Total loss (multi-task learning)
    L_total = L_click + λ·L_pop                      # λ는 balancing hyperparameter

    # Backward pass
    L_total.backward()
    optimizer.step()
```

**핵심:**
- L_pop는 **auxiliary loss**로서 News Encoder의 학습을 guide
- L_click이 main task, L_pop는 better representation 학습을 위한 보조 task
- λ 값으로 두 loss의 균형 조절 (일반적으로 λ ∈ [0.01, 0.1])

**Note:** f_j와 p_j는 이후 I4 Matching 단계에서 독립적으로 사용되므로, **별도의 가중치나 scaling 없이 순수하게 유지**

---

### 5.3 Component (I2): Hot-to-Warm User Modeling

**목적:** Challenge 2 (C2) 해결 - Hot user의 분산된 관심사를 정교하게 처리

#### Module 1: Candidate-guided News Selection

**목표:**
Hot user의 분산된 관심사 중 후보 뉴스와 관련 있는 클릭 뉴스만 선택

**상황:**
- 유저가 N(e.g., 100)개 뉴스기사를 클릭했다면
- 그 중에 후보뉴스 c와 더 관련있는 K개의 클릭뉴스를 selection
  - K 설정 옵션:
    - Warm user의 최대 클릭 수 (=상위 M% 클릭 수)
    - 단순하게 상위 K% (e.g., 30개)

**Selection 방법:**

**(1) Target-aware Attention:**

**설계 방침: 후보뉴스의 뉴스 임베딩만 사용**

```python
# Input
f_c: 후보뉴스의 popularity-free representation ∈ ℝ^d
{f_1, f_2, ..., f_N}: 클릭뉴스들의 popularity-free representations (N개)

# Attention 계산
Q = W_Q · f_c                              # Query: (d,) - 후보뉴스 query 벡터
K_j = W_K · f_j                            # Key: (d,) - j번째 클릭뉴스 key 벡터 (for j=1..N)
V_j = W_V · f_j                            # Value: (d,) - j번째 클릭뉴스 value 벡터 (for j=1..N)

# Attention scores (각 클릭뉴스마다 계산)
e_j = Q^T · K_j / sqrt(d)                  # scalar - j번째 클릭뉴스와의 유사도
α_j = exp(e_j) / Σ_k exp(e_k)              # scalar - j번째 클릭뉴스의 attention weight

# α의 차원 및 특성
# - α_j: scalar (단일 값) - j번째 클릭뉴스의 attention weight
# - α = {α_1, α_2, ..., α_N}: (N,) 벡터 - 모든 클릭뉴스의 attention weights
# - 제약: Σ_{j=1}^{N} α_j = 1 (softmax normalization)
```

**설계 이유:**
- f_c와 f_j는 이미 content semantic 정보를 담고 있음
- 유사한 콘텐츠의 뉴스는 높은 attention score를 받음
- 추후 확장: 토픽 임베딩 추가 가능 (Q = [f_c ; topic_emb_c])

**(2) Top-K Selection & Reweighting:**

**필수 기능**으로, Hot user의 분산된 관심사 중 후보 뉴스와 가장 관련 있는 K개만 강하게 반영

```python
# Step 1: Top-K 선택
# α = {α_1, α_2, ..., α_N}에서 상위 K개의 인덱스 선택
top_k_indices = argsort(α, descending=True)[:K]  # K개 인덱스

# Step 2: Reweighting
# Top-K에 포함되지 않은 클릭뉴스의 attention weight를 감소
α_j_reweighted = α_j           if j ∈ top_k_indices
α_j_reweighted = ε * α_j       otherwise

# Step 3: Re-normalization (optional, 실험적으로 결정)
α_j_final = α_j_reweighted / Σ_k α_k_reweighted  # 다시 합이 1이 되도록
```

**Parameters:**
- **K**: Top-K 개수 (default: **30**)
  - Hot user의 분산된 관심사 중 상위 30개 뉴스만 선택
  - 옵션: Warm user의 평균 클릭 수 기반으로 동적 조정 가능
- **ε**: Reweighting factor (default: **0.01**)
  - ε=0.01: Top-K 외 뉴스는 1% weight만 유지
  - ε=0: Top-K 외 뉴스는 완전히 무시
  - ε=1: Reweighting 없음 (모든 뉴스 동일 weight)

**의의:**
- Hot user의 100개 클릭뉴스 중 후보 뉴스와 무관한 70개는 거의 무시 (ε=0.01)
- 후보 뉴스와 관련 있는 30개만 강하게 반영 → 분산된 관심사 문제 해결

**(3) Gated Residual Connection:**

**참고: LIME Section 3.4**

최종 임베딩 구성:
```
r_j = g_i ⊙ (α_j · f_j) + (1 − g_i) ⊙ f_j
```

Gate 계산:
```
g_i = σ(W_g · f_j + α_j)
```

**gi 기반 r_j 메커니즘 상세 설명:**

**1. Gate gi의 역할:**
- **gi ∈ [0, 1]**: sigmoid 출력으로 0과 1 사이 값
- **gi가 1에 가까울수록**: attention-weighted 정보 (α_j · f_j) 많이 반영
- **gi가 0에 가까울수록**: 원본 정보 (f_j) 많이 반영

**2. Gate는 어떻게 결정되는가?**
```
g_i = σ(W_g · f_j + α_j)
```
- **W_g · f_j**: 뉴스 콘텐츠 자체의 특성을 학습
  - 어떤 타입의 뉴스는 attention을 많이 받아야 하고, 어떤 뉴스는 적게 받아야 함
- **+ α_j (bias term)**: attention score가 높으면 gate도 열리기 쉽게
  - α_j가 크면 → g_i도 커지는 경향 → attention 정보 많이 반영
  - LIME과의 차이: LIME은 bias 없이 W_g · f_j만 사용

**3. 최종 r_j 도출 과정:**
```
r_j = g_i ⊙ (α_j · f_j) + (1 − g_i) ⊙ f_j
    = g_i · α_j · f_j + (1 - g_i) · f_j
    = [g_i · α_j + (1 - g_i)] · f_j
```

**경우 분석:**
- **Case 1: α_j = 0.1 (낮은 attention), g_i = 0.2**
  - r_j = 0.2 × 0.1 × f_j + 0.8 × f_j = 0.82 × f_j
  - 원본에 가까움 (attention 거의 무시)

- **Case 2: α_j = 0.9 (높은 attention), g_i = 0.8**
  - r_j = 0.8 × 0.9 × f_j + 0.2 × f_j = 0.92 × f_j
  - attention 크게 반영

- **Case 3: α_j = 0.9 (높은 attention), g_i = 0.2 (gate 닫힘)**
  - r_j = 0.2 × 0.9 × f_j + 0.8 × f_j = 0.98 × f_j
  - Gate가 attention의 영향을 제어 (overfitting 방지)

**4. 왜 이런 메커니즘이 필요한가?**
- **Residual connection**: 원본 정보 보존으로 gradient flow 개선
- **Gated mechanism**: 모델이 학습을 통해 attention을 얼마나 신뢰할지 결정
- **Adaptive weighting**: 각 뉴스마다 다른 가중치 적용 가능

**Note:** **f_j (popularity-free news rep)**를 직접 사용하여 user encoder의 input 구성

---

**5. W_g 학습 메커니즘 상세 설명:**

**핵심 질문:** W_g는 어떻게 학습되고, 어떤 Loss를 통해 업데이트되는가?

**답변:** W_g는 **학습 가능한 파라미터**이며, **L_click (클릭 예측 손실)**을 통해 학습됨

**W_g가 연결된 Computational Graph:**

```
g_i = σ(W_g · f_j + α_j)  ← W_g가 여기 사용됨
      ↓
r_j = g_i ⊙ (α_j · f_j) + (1 - g_i) ⊙ f_j
      ↓
u = UserEncoder(r_j)  ← User representation 생성
      ↓
score = σ(α · (u·f_c - β·mean(p_j·p_c)))  ← Contrastive matching score
      ↓
L_click = -log(score)  ← 클릭 예측 손실 (positive sample)
      ↓
L_total = L_click + λ·L_pop
```

**Gradient Flow (Backpropagation):**

`L_total.backward()` 호출 시, PyTorch autograd가 다음과 같이 gradient를 계산:

```
∂L_total/∂W_g 계산 과정:

L_click
  ↓ ∂L_click/∂score
score (클릭 예측 점수)
  ↓ ∂score/∂u
u (user representation)
  ↓ ∂u/∂r_j  (User Encoder의 파라미터들을 통과)
r_j (= g_i ⊙ (α_j · f_j) + (1 - g_i) ⊙ f_j)
  ↓ ∂r_j/∂g_i = α_j · f_j - f_j = f_j · (α_j - 1)
g_i (= σ(W_g · f_j + α_j))
  ↓ ∂g_i/∂(W_g·f_j) = σ'(W_g · f_j + α_j) = g_i · (1 - g_i)
  ↓ ∂(W_g·f_j)/∂W_g = f_j^T
W_g
  ↓
∂L_click/∂W_g = (chain rule로 계산된 gradient)
```

**최종 gradient:**
```
∂L_click/∂W_g = ∂L_click/∂score · ∂score/∂u · ∂u/∂r_j · ∂r_j/∂g_i · ∂g_i/∂W_g
```

**파라미터 업데이트:**
```
W_g ← W_g - lr · ∂L_click/∂W_g
```

**학습 목표:**

W_g는 다음 질문에 답하도록 학습됨:
- **"각 클릭뉴스의 attention weight를 얼마나 신뢰해야 클릭 예측을 잘 할 수 있는가?"**

**학습되는 패턴 (뉴스 타입별 attention 신뢰도):**

```python
# 예시 1: 스포츠 뉴스 (Hot user가 많이 클릭하지만 실제 추천과 무관한 경우가 많음)
#
# 학습 데이터에서 발견된 패턴:
# - Hot user가 스포츠 뉴스를 많이 클릭 → α_j(스포츠) 높음
# - 하지만 실제 후보뉴스 클릭과 무관 → L_click 증가
#
# W_g의 학습 방향:
# - W_g · f_j(스포츠) → 낮은 값으로 학습
# - g_i = σ(낮은값 + α_j) → g_i 작아짐 (예: 0.3)
# - r_j ≈ 0.3 × (α_j × f_j) + 0.7 × f_j ≈ f_j (원본에 가까움)
# - 결과: attention을 거의 무시 → L_click 감소 (클릭 예측 향상)

# 예시 2: 건강 뉴스 (Hot user가 적당히 클릭하지만 실제 추천과 연관 높음)
#
# 학습 데이터에서 발견된 패턴:
# - Hot user가 건강 뉴스를 적당히 클릭 → α_j(건강) 중간
# - 실제 후보뉴스 클릭과 연관 높음 → L_click 감소
#
# W_g의 학습 방향:
# - W_g · f_j(건강) → 높은 값으로 학습
# - g_i = σ(높은값 + α_j) → g_i 커짐 (예: 0.8)
# - r_j ≈ 0.8 × (α_j × f_j) + 0.2 × f_j ≈ α_j × f_j (attention 많이 반영)
# - 결과: attention을 많이 반영 → L_click 감소 (클릭 예측 향상)
```

**W_g 학습의 의의:**

1. **Attention 신뢰도 자동 학습**:
   - 초기화 시: W_g 랜덤 초기화 → 모든 뉴스에 대해 g_i ≈ 0.5 (동일한 attention 반영)
   - 학습 후: W_g가 뉴스 타입별로 "이 타입의 뉴스는 attention을 얼마나 신뢰해야 하는가" 학습

2. **Overfitting 방지**:
   - Attention이 높아도 (α_j 큰 값) 실제 클릭과 무관하면 g_i를 낮춰서 무시
   - "Attention이 틀릴 수 있다"는 것을 학습을 통해 자동으로 판단

3. **Adaptive Gating**:
   - 각 뉴스마다 콘텐츠 특성 (W_g · f_j)과 attention score (α_j)를 모두 고려
   - 단순히 α_j만 사용하는 것보다 robust한 user representation 생성

**PyTorch 구현 예시:**

```python
class CandidateGuidedNewsSelector(nn.Module):
    """
    W_g를 포함한 Candidate-guided News Selection 모듈
    """
    def __init__(self, d):
        super().__init__()
        # W_g: 학습 가능한 파라미터
        # Input 차원: d (뉴스 representation)
        # Output 차원: 1 (scalar gate value)
        self.W_g = nn.Linear(d, 1, bias=False)
        # ↑ W_g.weight: (1, d) 형태의 학습 가능한 파라미터

    def forward(self, f_j, α_j):
        """
        Args:
            f_j: (N, d) - 클릭뉴스들의 popularity-free representations
            α_j: (N,) - attention weights

        Returns:
            r_j: (N, d) - gated residual connection 결과
        """
        # Gate 계산: g_i = σ(W_g · f_j + α_j)
        # Input: f_j (N, d)
        # W_g · f_j: (N, d) @ (d, 1)^T = (N, 1)
        g_logits = self.W_g(f_j).squeeze(-1)  # (N,)

        # Bias term 추가
        g_logits = g_logits + α_j  # (N,)

        # Sigmoid activation
        g_i = torch.sigmoid(g_logits)  # (N,) - 각 뉴스의 gate value

        # Gated residual connection
        g_i_expanded = g_i.unsqueeze(-1)  # (N, 1)
        α_j_expanded = α_j.unsqueeze(-1)  # (N, 1)

        r_j = g_i_expanded * (α_j_expanded * f_j) + (1 - g_i_expanded) * f_j
        # r_j: (N, d)

        return r_j

# Training Loop에서 W_g 학습
optimizer = torch.optim.Adam([
    {'params': news_encoder.parameters()},
    {'params': disentangler.parameters()},
    {'params': selector.parameters()},  # ← W_g가 여기 포함됨!
    {'params': user_encoder.parameters()},
])

for batch in dataloader:
    # Forward pass
    h_j = news_encoder(news_title_j)
    f_j, p_j = disentangler(h_j)
    f_c, p_c = disentangler(h_c)

    # Attention 계산
    α_j = compute_attention(f_c, f_j)  # (N,)

    # r_j 계산 (W_g 사용)
    r_j = selector(f_j, α_j)  # 내부에서 g_i = σ(W_g · f_j + α_j) 계산

    # User representation
    u = user_encoder(r_j)

    # Contrastive matching
    score = contrastive_matching(u, f_c, p_c, p_j)

    # Loss 계산
    L_click = -torch.log(score)  # positive sample
    L_total = L_click + λ * L_pop

    # Backward pass
    optimizer.zero_grad()
    L_total.backward()
    # ↑ 이 시점에 ∂L_click/∂W_g가 자동 계산되어 W_g.grad에 저장됨!

    # 파라미터 업데이트 (W_g 포함)
    optimizer.step()
    # W_g ← W_g - lr · ∂L_click/∂W_g
```

**비유로 이해하기:**

- **α_j (Attention)**: "후보뉴스와 이 클릭뉴스가 유사합니다!" (Attention의 주장)
- **g_i (Gate)**: "그 주장을 얼마나 믿을 것인가?" (Gate의 판단)
- **W_g (학습 파라미터)**: "과거 경험(학습 데이터)을 통해 각 뉴스 타입별로 Attention을 얼마나 신뢰할지 학습한 지식"

**핵심:**
- W_g는 **L_click을 minimize하는 방향으로 학습**됨
- 클릭 예측을 잘 하기 위해 **뉴스 타입별 attention 신뢰도**를 자동으로 학습
- PyTorch autograd가 모든 gradient 계산 자동 처리

---

**6. r_j 도출 전체 파이프라인 구현:**

**목적:** Top-K Selection → Gated Residual Connection을 연속적으로 수행하여 최종 **r_j** 도출

**프로세스 요약:**
1. **Target-aware Attention 계산** → α (N,) - 각 클릭뉴스의 attention weight
2. **Top-K Selection & Reweighting** → α_final (N,) - reweighted & re-normalized
3. **Gated Residual Connection** → r_j (N, d) - 최종 클릭뉴스 representation
4. **User Encoder** → u (d,) - 최종 유저 representation

**PyTorch 구현 코드:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CandidateGuidedNewsSelector(nn.Module):
    """
    목적: Hot user의 분산된 관심사 중 후보뉴스와 관련있는 클릭뉴스만 선택하여
          최종 r_j를 도출하는 모듈

    입력:
        - f_c: 후보뉴스의 popularity-free representation (d,)
        - f_j: 클릭뉴스들의 popularity-free representations (N, d)

    출력:
        - r_j: reweighted된 클릭뉴스 representations (N, d)
        - α_final: 최종 attention weights (N,) - monitoring용

    의의:
        - Hot user의 100개 클릭뉴스 중 후보뉴스와 가장 관련있는 K개만 강하게 반영
        - 나머지 뉴스는 ε만큼만 weight 유지
        - W_g 학습을 통해 attention 신뢰도 자동 조절
    """

    def __init__(self, d, K=30, epsilon=0.01):
        """
        Args:
            d: representation 차원
            K: Top-K 개수 (default: 30)
            epsilon: reweighting factor (default: 0.01)
        """
        super().__init__()
        self.d = d
        self.K = K
        self.epsilon = epsilon

        # (1) Target-aware Attention을 위한 파라미터
        self.W_Q = nn.Linear(d, d, bias=False)  # Query projection
        self.W_K = nn.Linear(d, d, bias=False)  # Key projection

        # (3) Gated Residual Connection을 위한 파라미터
        self.W_g = nn.Linear(d, 1, bias=False)  # Gate projection

    def forward(self, f_c, f_j):
        """
        Args:
            f_c: (d,) - 후보뉴스의 popularity-free representation
            f_j: (N, d) - 클릭뉴스들의 popularity-free representations

        Returns:
            r_j: (N, d) - 최종 클릭뉴스 representations
            α_final: (N,) - 최종 attention weights
        """
        N, d = f_j.shape

        # ===================================================================
        # Step 1: Target-aware Attention 계산
        # ===================================================================

        # Query: 후보뉴스 f_c로부터 계산
        # Input: f_c (d,)
        # Output: Q (d,)
        Q = self.W_Q(f_c)  # (d,)

        # Key: 각 클릭뉴스 f_j로부터 계산
        # Input: f_j (N, d)
        # Output: K_j (N, d)
        K_j = self.W_K(f_j)  # (N, d)

        # Attention score 계산: e_j = Q^T · K_j / sqrt(d)
        # Input: Q (d,), K_j (N, d)
        # Output: e_j (N,) - 각 클릭뉴스와 후보뉴스의 유사도
        e_j = torch.matmul(K_j, Q) / torch.sqrt(torch.tensor(d, dtype=torch.float32))  # (N,)

        # Softmax normalization: α_j = exp(e_j) / Σ_k exp(e_k)
        # Input: e_j (N,)
        # Output: α (N,) - Σα_j = 1 (softmax property)
        α = F.softmax(e_j, dim=0)  # (N,)
        # α_j: scalar - j번째 클릭뉴스의 attention weight
        # α: (N,) 벡터 - 모든 클릭뉴스의 attention weights

        # ===================================================================
        # Step 2: Top-K Selection & Reweighting
        # ===================================================================

        # (2-1) Top-K 선택
        # Input: α (N,)
        # Output: top_k_indices (K,) - 상위 K개 인덱스
        top_k_values, top_k_indices = torch.topk(α, k=min(self.K, N), largest=True)

        # (2-2) Reweighting
        # Top-K에 포함되지 않은 클릭뉴스는 ε만큼만 weight 유지
        # Input: α (N,)
        # Output: α_reweighted (N,)
        α_reweighted = torch.ones_like(α) * self.epsilon * α  # 모두 ε*α로 초기화
        α_reweighted[top_k_indices] = α[top_k_indices]  # Top-K는 원래 α 유지

        # (2-3) Re-normalization
        # 다시 합이 1이 되도록 정규화
        # Input: α_reweighted (N,)
        # Output: α_final (N,) - Σα_final = 1
        α_final = α_reweighted / torch.sum(α_reweighted)  # (N,)

        # ===================================================================
        # Step 3: Gated Residual Connection
        # ===================================================================

        # Gate 계산: g_i = σ(W_g · f_j + α_final)
        # Input: f_j (N, d), α_final (N,)
        # Output: g_i (N,) - 각 클릭뉴스의 gate value

        # W_g · f_j: (N, d) @ (d, 1) → (N, 1) → (N,)
        g_logits = self.W_g(f_j).squeeze(-1)  # (N,)

        # + α_final (bias term)
        g_logits = g_logits + α_final  # (N,)

        # Sigmoid activation: g_i ∈ [0, 1]
        g_i = torch.sigmoid(g_logits)  # (N,)

        # 최종 r_j 계산: r_j = g_i ⊙ (α_final · f_j) + (1 - g_i) ⊙ f_j
        # Input:
        #   - g_i (N,)
        #   - α_final (N,)
        #   - f_j (N, d)
        # Output: r_j (N, d)

        # Element-wise 계산을 위해 차원 확장
        g_i_expanded = g_i.unsqueeze(-1)  # (N, 1)
        α_final_expanded = α_final.unsqueeze(-1)  # (N, 1)

        # r_j = g_i * (α_final * f_j) + (1 - g_i) * f_j
        r_j = g_i_expanded * (α_final_expanded * f_j) + (1 - g_i_expanded) * f_j  # (N, d)

        # 의의:
        # - α_final: Top-K 외 뉴스는 거의 무시 (ε=0.01)
        # - g_i: 각 뉴스마다 attention을 얼마나 신뢰할지 학습
        # - r_j: 최종 클릭뉴스 representation (후보뉴스 기반 reweighted)

        return r_j, α_final


# ===================================================================
# 사용 예시 1: CandidateGuidedNewsSelector 단독 사용
# ===================================================================
def example_selector_usage():
    """
    CandidateGuidedNewsSelector 사용 예시
    """
    d = 256  # representation 차원
    N = 100  # 클릭뉴스 개수 (Hot user)
    K = 30   # Top-K 개수
    epsilon = 0.01  # reweighting factor

    # Model 정의
    selector = CandidateGuidedNewsSelector(d=d, K=K, epsilon=epsilon)

    # Dummy data (실제로는 NewsEncoder + PopularityDisentangler에서 생성)
    # f_c: 후보뉴스의 popularity-free representation
    # f_j: 클릭뉴스들의 popularity-free representations
    f_c = torch.randn(d)      # (d,)
    f_j = torch.randn(N, d)   # (N, d)

    # Forward pass: r_j 도출
    # Input: f_c (d,), f_j (N, d)
    # Output: r_j (N, d), α_final (N,)
    r_j, α_final = selector(f_c, f_j)

    print(f"Input:")
    print(f"  f_c: {f_c.shape}")  # (256,)
    print(f"  f_j: {f_j.shape}")  # (100, 256)
    print(f"\nOutput:")
    print(f"  r_j: {r_j.shape}")  # (100, 256)
    print(f"  α_final: {α_final.shape}")  # (100,)
    print(f"  α_final sum: {α_final.sum():.4f}")  # 1.0000 (re-normalized)
    print(f"  Top-K indices: {torch.topk(α_final, k=K).indices}")


# ===================================================================
# 사용 예시 2: Hot-to-Warm User Modeling 전체 파이프라인
# ===================================================================
class HotToWarmUserModeling(nn.Module):
    """
    Component (I2): Hot-to-Warm User Modeling 전체 파이프라인

    입력:
        - f_c: 후보뉴스 (d,)
        - f_j: 클릭뉴스들 (N, d)

    출력:
        - u: 유저 representation (d,)

    의의:
        - Hot user의 분산된 관심사를 정교하게 처리
        - 후보뉴스 기반으로 관련 있는 클릭뉴스만 선택
        - 최종 유저 representation 생성
    """

    def __init__(self, d, K=30, epsilon=0.01, user_encoder=None):
        super().__init__()
        # Module 1: Candidate-guided News Selection
        self.selector = CandidateGuidedNewsSelector(d=d, K=K, epsilon=epsilon)

        # Module 2: User Encoder (plug-in)
        self.user_encoder = user_encoder if user_encoder is not None else nn.Identity()

    def forward(self, f_c, f_j):
        """
        Args:
            f_c: (d,) - 후보뉴스의 popularity-free representation
            f_j: (N, d) - 클릭뉴스들의 popularity-free representations

        Returns:
            u: (d,) - user representation
        """
        # Step 1-3: r_j 도출 (Top-K Selection → Gated Residual Connection)
        # Input: f_c (d,), f_j (N, d)
        # Output: r_j (N, d)
        r_j, α_final = self.selector(f_c, f_j)  # r_j: (N, d)

        # Step 4: User Encoder
        # Input: r_j (N, d) - reweighted 클릭뉴스 representations
        # Output: u (d,) - user representation
        u = self.user_encoder(r_j)  # (d,)

        return u
```

**핵심 프로세스 요약:**

| Step | 작업 | 입력 | 출력 | 의의 |
|------|------|------|------|------|
| **1** | Target-aware Attention | f_c (d,), f_j (N, d) | α (N,) | 후보뉴스와 각 클릭뉴스의 유사도 계산 |
| **2** | Top-K Selection & Reweighting | α (N,) | α_final (N,) | Top-30만 강하게 반영, 나머지 70개는 1% weight |
| **3** | Gated Residual Connection | f_j (N, d), α_final (N,) | r_j (N, d) | Attention을 얼마나 신뢰할지 학습하여 최종 r_j 도출 |
| **4** | User Encoder | r_j (N, d) | u (d,) | 최종 유저 representation 생성 |

**데이터 흐름 예시 (Hot user: N=100, K=30, ε=0.01):**

```
f_c (256,) + f_j (100, 256)
    ↓ [Step 1: Attention]
α (100,) - 모든 클릭뉴스의 attention weights, Σα = 1
    ↓ [Step 2-1: Top-K Selection]
top_k_indices (30,) - 상위 30개 인덱스
    ↓ [Step 2-2: Reweighting]
α_reweighted (100,) - Top-30: α 유지, 나머지 70: 0.01*α
    ↓ [Step 2-3: Re-normalization]
α_final (100,) - Σα_final = 1
    ↓ [Step 3: Gated Residual Connection]
r_j (100, 256) - 최종 클릭뉴스 representations
    ↓ [Step 4: User Encoder]
u (256,) - 최종 유저 representation
```

**핵심:**
- **r_j가 User Encoder의 실제 input**으로 사용됨
- Hot user의 100개 클릭뉴스 중 **후보뉴스와 가장 관련있는 30개만 강하게 반영**
- 나머지 70개는 **1% weight만 유지** → 분산된 관심사 문제 해결
- **W_g 학습**을 통해 각 뉴스 타입별로 attention 신뢰도 자동 조절

---

#### Module 2: User Encoder

**설계:**
- **Plug-in design:** 기존 방법들 중 유저인코더가 있는 어떤 방법도 갈아끼워서 사용 가능
- **Input:** r_j (후보뉴스 기반으로 reweighted된 **popularity-free 클릭뉴스 임베딩**들)
- **Output:** u (유저 임베딩)

---

### 5.4 Component (I3): Topic-wise Popularity Modeling

**목적:** Challenge 3 (C3) 해결 - 토픽별 상대적 인기도 고려

#### Module 1: Topic-wise Popularity Normalization

**목표:** 같은 클릭 수라도 토픽에 따라 다른 인기도를 학습

---

**PENR [CIKM'21] 기반 인기도 계산 상세 설명:**

**PENR (Popularity-Enhanced News Recommendation)** 논문에서는 뉴스의 인기도를 **클릭 수 기반으로 [0, 1] 범위로 정규화**하여 계산합니다.

**1. 기본 개념:**

```python
# PENR의 인기도 계산 방식
popularity_j = click_count_j / max(all_click_counts)
```

**각 변수의 의미:**
- **click_count_j**: 뉴스 j가 받은 총 클릭 수 (impression 대비가 아닌 절대 클릭 수)
- **max(all_click_counts)**: 전체 뉴스 중 가장 많은 클릭을 받은 뉴스의 클릭 수
- **popularity_j**: 뉴스 j의 정규화된 인기도 ∈ [0, 1]

**2. 시간 윈도우 및 데이터 수집:**

```python
# 시간 윈도우 설정 (PENR 기준)
time_window = 12  # hours (뉴스 발행 후 12시간)

# 각 뉴스의 클릭 수 수집
for news_j in all_news:
    # 뉴스 j 발행 시각: publish_time_j
    # 윈도우 종료 시각: publish_time_j + 12 hours
    click_count_j = count_clicks(news_j, start=publish_time_j, end=publish_time_j + 12h)
```

**의의:**
- 12시간 윈도우: 뉴스의 생명주기(lifetime)가 짧기 때문에 초기 12시간 내 클릭 수로 인기도 측정
- 모든 뉴스에 대해 동일한 시간 윈도우 적용 → 공정한 비교

**3. 정규화 과정 상세:**

```python
# Step 1: 전체 뉴스의 클릭 수 수집
all_news = get_all_news_in_dataset()
click_counts = [count_clicks(news, time_window=12) for news in all_news]

# Step 2: 최대 클릭 수 계산
max_clicks = max(click_counts)
# 예: max_clicks = 50,000 (가장 인기 많은 뉴스가 50,000 클릭을 받음)

# Step 3: 각 뉴스의 인기도 정규화 [0, 1]
for news_j in all_news:
    popularity_j = click_count_j / max_clicks
    # popularity_j ∈ [0, 1]
    # - 0에 가까울수록: 클릭을 거의 받지 못한 뉴스
    # - 1에 가까울수록: 많은 클릭을 받은 인기 뉴스
    # - 1.0: 최대 클릭을 받은 뉴스 (가장 인기 많은 뉴스)
```

**4. 구체적인 계산 예시:**

| 뉴스 ID | 토픽 | 클릭 수 (12h) | max_clicks | PENR popularity |
|--------|------|--------------|-----------|-----------------|
| News A | Sports | 50,000 | 50,000 | 50,000 / 50,000 = **1.0** |
| News B | Entertainment | 25,000 | 50,000 | 25,000 / 50,000 = **0.5** |
| News C | Health | 500 | 50,000 | 500 / 50,000 = **0.01** |
| News D | Sports | 500 | 50,000 | 500 / 50,000 = **0.01** |
| News E | Politics | 10,000 | 50,000 | 10,000 / 50,000 = **0.2** |

**관찰:**
- News C (Health, 500 clicks)와 News D (Sports, 500 clicks) 모두 popularity = **0.01** (동일)
- 하지만 토픽별 맥락을 고려하면:
  - **Health 토픽**: 평균적으로 클릭이 적음 → 500 clicks는 **상대적으로 많은 클릭**
  - **Sports 토픽**: 평균적으로 클릭이 많음 → 500 clicks는 **상대적으로 적은 클릭**
- PENR은 이러한 토픽별 차이를 고려하지 못함 → **POPCORN의 개선점**

**5. PENR 인기도의 한계:**

```python
# PENR 방식 (토픽 무시)
popularity_health_500 = 500 / 50,000 = 0.01
popularity_sports_500 = 500 / 50,000 = 0.01
# → 두 뉴스 모두 동일한 인기도 (토픽 맥락 무시)
```

**문제점:**
- 동일한 클릭 수라도 **토픽에 따라 의미가 다름**
- Health 뉴스 500 clicks: 해당 토픽 내에서는 **매우 인기 있는 뉴스**
- Sports 뉴스 500 clicks: 해당 토픽 내에서는 **인기 없는 뉴스**
- PENR은 이러한 **상대적 인기도(relative popularity)**를 반영하지 못함

---

**POPCORN의 차이점: 토픽별 재정규화 (Topic-wise Re-normalization)**

PENR과 달리, **토픽별로 상대적 인기도를 다시 normalization**하여 위 한계를 해결합니다.

**전체 프로세스:**

```python
# ===================================================================
# Step 1: PENR 방식으로 기본 인기도 계산 (Raw Popularity)
# ===================================================================

all_news = get_all_news_in_dataset()
max_clicks = max([count_clicks(news, time_window=12) for news in all_news])

for news_j in all_news:
    click_count_j = count_clicks(news_j, time_window=12)
    raw_popularity_j = click_count_j / max_clicks
    # raw_popularity_j ∈ [0, 1]

# ===================================================================
# Step 2: 토픽별 평균 인기도 계산 (Topic-wise Average)
# ===================================================================

# 각 토픽별로 평균 인기도 계산
topics = get_unique_topics(all_news)  # e.g., ['Sports', 'Health', 'Entertainment', ...]

topic_avg_popularity = {}
for topic_t in topics:
    # 해당 토픽의 모든 뉴스 가져오기
    news_in_topic_t = [news for news in all_news if get_topic(news) == topic_t]

    # 해당 토픽 뉴스들의 평균 raw_popularity 계산
    avg_popularity_t = mean([raw_popularity[news] for news in news_in_topic_t])

    topic_avg_popularity[topic_t] = avg_popularity_t

# 예시 결과:
# topic_avg_popularity = {
#     'Sports': 0.5,        # Sports 뉴스는 평균적으로 인기도 0.5
#     'Health': 0.06,       # Health 뉴스는 평균적으로 인기도 0.06
#     'Entertainment': 0.4,
#     'Politics': 0.15,
#     ...
# }

# ===================================================================
# Step 3: 토픽별 상대적 인기도로 재정규화 (POPCORN의 핵심!)
# ===================================================================

for news_j in all_news:
    topic_j = get_topic(news_j)
    raw_popularity_j = raw_popularity[news_j]
    avg_popularity_j = topic_avg_popularity[topic_j]

    # 토픽 평균으로 나눔 → 토픽 내 상대적 인기도
    normalized_popularity_j = raw_popularity_j / avg_popularity_j
    # normalized_popularity_j: 토픽 내에서 평균 대비 몇 배 인기 있는지
```

**구체적인 계산 예시:**

| 뉴스 ID | 토픽 | 클릭 수 | Raw Pop (PENR) | 토픽 평균 | Normalized Pop (POPCORN) | 의미 |
|--------|------|---------|---------------|----------|------------------------|------|
| News C | Health | 500 | 0.01 | 0.06 | 0.01 / 0.06 = **0.167** | 토픽 평균보다 낮음 |
| News D | Sports | 500 | 0.01 | 0.5 | 0.01 / 0.5 = **0.02** | 토픽 평균보다 매우 낮음 |
| News F | Health | 3,000 | 0.06 | 0.06 | 0.06 / 0.06 = **1.0** | 토픽 평균 수준 |
| News G | Health | 6,000 | 0.12 | 0.06 | 0.12 / 0.06 = **2.0** | 토픽 평균의 2배 (매우 인기) |
| News A | Sports | 50,000 | 1.0 | 0.5 | 1.0 / 0.5 = **2.0** | 토픽 평균의 2배 (매우 인기) |

**POPCORN Normalized Popularity 해석:**
- **< 1.0**: 해당 토픽 내에서 평균보다 인기가 **낮음**
- **= 1.0**: 해당 토픽 내에서 **평균 수준**의 인기
- **> 1.0**: 해당 토픽 내에서 평균보다 인기가 **높음**
- **예: 2.0**: 해당 토픽 평균 인기도의 **2배**

**PENR vs POPCORN 비교:**

| 뉴스 | PENR (토픽 무시) | POPCORN (토픽별 정규화) | 차이점 |
|------|-----------------|----------------------|--------|
| Health 500 clicks | 0.01 (낮음) | 0.167 (토픽 내 평균 이하) | 토픽 맥락 반영 |
| Sports 500 clicks | 0.01 (낮음) | 0.02 (토픽 내 매우 낮음) | 토픽 맥락 반영 |
| Health 6,000 clicks | 0.12 (중간) | 2.0 (토픽 내 매우 높음) | 상대적 인기도 강조 |

**최종 공식 (POPCORN):**

```python
# Topic-wise Normalized Popularity
normalized_popularity_j = raw_popularity_j / avg_popularity_topic(j)

# where:
# - raw_popularity_j = click_count_j / max(all_click_counts)  (PENR 방식)
# - avg_popularity_topic(j) = mean(raw_popularity for all news in same topic)
```

**핵심 차이점:**
- **PENR**: 모든 뉴스를 단일 척도로 비교 → 토픽 맥락 무시
- **POPCORN**: 토픽 내 상대적 위치로 비교 → 토픽별 attractiveness 차이 반영

**의의:**
1. **상대적 인기도 반영**: 동일한 클릭 수라도 토픽에 따라 다른 의미
2. **토픽별 attractiveness 고려**:
   - High-attractiveness topics (Sports, Entertainment): 높은 기준 적용
   - Low-attractiveness topics (Health, Finance): 낮은 기준 적용
3. **C3 Challenge 해결**: 토픽별 상대적 인기도를 명시적으로 모델링

**실제 구현 예시 코드:**

```python
import numpy as np
from collections import defaultdict

def compute_topic_wise_popularity(news_data, time_window=12):
    """
    POPCORN의 Topic-wise Normalized Popularity 계산

    입력:
        - news_data: List of news items
            [
                {'id': 1, 'topic': 'Sports', 'clicks': 50000, ...},
                {'id': 2, 'topic': 'Health', 'clicks': 500, ...},
                ...
            ]
        - time_window: 클릭 수를 세는 시간 윈도우 (hours)

    출력:
        - popularity_dict: {news_id: normalized_popularity}
    """

    # ===================================================================
    # Step 1: PENR 방식으로 Raw Popularity 계산
    # ===================================================================
    max_clicks = max([news['clicks'] for news in news_data])

    raw_popularity = {}
    for news in news_data:
        news_id = news['id']
        click_count = news['clicks']
        raw_popularity[news_id] = click_count / max_clicks

    # ===================================================================
    # Step 2: 토픽별 평균 인기도 계산
    # ===================================================================
    topic_popularity = defaultdict(list)

    for news in news_data:
        topic = news['topic']
        news_id = news['id']
        topic_popularity[topic].append(raw_popularity[news_id])

    # 각 토픽의 평균 인기도 계산
    topic_avg_popularity = {}
    for topic, popularities in topic_popularity.items():
        topic_avg_popularity[topic] = np.mean(popularities)

    # ===================================================================
    # Step 3: 토픽별 상대적 인기도로 재정규화
    # ===================================================================
    normalized_popularity = {}

    for news in news_data:
        news_id = news['id']
        topic = news['topic']
        raw_pop = raw_popularity[news_id]
        avg_pop = topic_avg_popularity[topic]

        # 토픽 평균으로 나눔
        normalized_popularity[news_id] = raw_pop / avg_pop

    return normalized_popularity, topic_avg_popularity


# 사용 예시
news_data = [
    {'id': 1, 'topic': 'Sports', 'clicks': 50000},
    {'id': 2, 'topic': 'Health', 'clicks': 500},
    {'id': 3, 'topic': 'Sports', 'clicks': 500},
    {'id': 4, 'topic': 'Health', 'clicks': 6000},
]

normalized_pop, topic_avg = compute_topic_wise_popularity(news_data)

print("PENR Raw Popularity:")
for news in news_data:
    raw = news['clicks'] / 50000
    print(f"  News {news['id']} ({news['topic']}): {raw:.4f}")

print("\nTopic Average Popularity:")
for topic, avg in topic_avg.items():
    print(f"  {topic}: {avg:.4f}")

print("\nPOPCORN Normalized Popularity:")
for news_id, norm_pop in normalized_pop.items():
    print(f"  News {news_id}: {norm_pop:.4f}")

# 출력:
# PENR Raw Popularity:
#   News 1 (Sports): 1.0000
#   News 2 (Health): 0.0100
#   News 3 (Sports): 0.0100
#   News 4 (Health): 0.1200
#
# Topic Average Popularity:
#   Sports: 0.5050
#   Health: 0.0650
#
# POPCORN Normalized Popularity:
#   News 1: 1.9802 (Sports 토픽 평균의 약 2배)
#   News 2: 0.1538 (Health 토픽 평균보다 낮음)
#   News 3: 0.0198 (Sports 토픽 평균보다 매우 낮음)
#   News 4: 1.8462 (Health 토픽 평균의 약 1.8배)
```

**Popularity Binning (10-class Classification):**

POPCORN에서는 이렇게 계산된 normalized_popularity를 **10개 클래스로 binning**하여 classification 문제로 변환합니다:

```python
def popularity_to_class(normalized_popularity):
    """
    Normalized popularity를 10개 클래스로 변환

    입력: normalized_popularity (float) - 토픽별 정규화된 인기도
    출력: class (int) - 0~9 중 하나
    """
    # Clipping: [0, 2] 범위로 제한 (대부분의 값이 이 범위에 분포)
    clipped = np.clip(normalized_popularity, 0.0, 2.0)

    # [0, 2] → [0, 10) 범위로 scaling
    scaled = clipped * 5  # [0, 10) 범위

    # 정수 클래스로 변환
    popularity_class = int(scaled)

    # 10 이상은 클래스 9로
    popularity_class = min(popularity_class, 9)

    return popularity_class

# 예시
print("Popularity Class:")
for news_id, norm_pop in normalized_pop.items():
    pop_class = popularity_to_class(norm_pop)
    print(f"  News {news_id}: {norm_pop:.4f} → class {pop_class}")

# 출력:
#   News 1: 1.9802 → class 9
#   News 2: 0.1538 → class 0
#   News 3: 0.0198 → class 0
#   News 4: 1.8462 → class 9
```

**Class 의미:**
- **Class 0**: 토픽 내 인기도 0.0~0.2 (매우 낮음)
- **Class 1**: 토픽 내 인기도 0.2~0.4
- **Class 2**: 토픽 내 인기도 0.4~0.6
- **Class 3**: 토픽 내 인기도 0.6~0.8
- **Class 4**: 토픽 내 인기도 0.8~1.0 (평균 수준)
- **Class 5**: 토픽 내 인기도 1.0~1.2 (평균 이상)
- **Class 6**: 토픽 내 인기도 1.2~1.4
- **Class 7**: 토픽 내 인기도 1.4~1.6
- **Class 8**: 토픽 내 인기도 1.6~1.8
- **Class 9**: 토픽 내 인기도 1.8~2.0+ (매우 높음)

---

#### Module 2: Topic-wise Popularity Prediction

**목적:** 토픽 정보를 활용하여 인기도를 예측하는 auxiliary task

**Architecture:**

**Input:**
```
[p_j or f_j ; topic_emb_j]
```
- `p_j or f_j`: popularity-aware news rep. or popularity-free news rep.
- `topic_emb_j = Dense([category_emb_j ; subcategory_emb_j])`

**Output:**
```
ŷ_jp or ŷ_jf (predicted popularity distribution) ∈ ℝ^10
```

**Training 시:**

1. **오차 계산 (Error Calculation):**
   - ŷ_jp와 y_j (or ŷ_jf와 y_j) 간의 **cross-entropy**로 오차를 구함
   - y_j: ground truth popularity class (0~9 중 하나)
   - ŷ_jp: predicted probability distribution

2. **목적함수 정의 (Loss Function):**
   - 오차값으로 **Loss (L_p, L_a)** 정의
   - L_p = -y_j log(ŷ_jp): p_j가 인기도를 잘 예측하도록
   - L_a = -1 / (y_j log(ỹ_jf)): f_j가 인기도를 예측하지 못하도록

3. **최적화 (Optimization):**
   - Loss를 미분하여 **Gradient** 구함
   - Backpropagation으로 News Encoder와 Popularity Disentangler 파라미터 업데이트

**Test 시:**
- Candidate news의 인기도를 예측하여 **auxiliary loss**로 사용

---

### 5.5 Component (I4): Popularity-aware Contrastive Interest Matching

**목적:** Challenge 1, 2, 3 통합 해결 - Interest와 Popularity를 Matching 단계에서 명시적으로 대조

**핵심 아이디어:**
- Representation 단계에서 f_j와 p_j를 독립적으로 유지
- Matching 단계에서 두 signal을 contrastive하게 비교하여 추천 결정

---

#### Input 준비

**Clicked News:**
```
News Encoder → h_j → Popularity Disentangler
    ├─→ f_j (clicked pop-free news rep)
    └─→ p_j (clicked pop-aware news rep)
```

**User:**
```
r_j (from f_j) → User Encoder → u (user rep)
```

**Candidate News:**
```
News Encoder → h_c → Popularity Disentangler
    ├─→ f_c (candidate pop-free news rep)
    └─→ p_c (candidate pop-aware news rep)
```

---

#### Two Matching Scores

**(1) Interest Matching: u · f_c**

**의미:**
- 유저의 "순수 관심사"가 후보뉴스의 "순수 컨텐츠"와 얼마나 유사한가?

**해석:**
- 높을수록 **interest signal**로 인한 클릭 가능성이 높음

---

**(2) Popularity Matching: mean(p_j · p_c)**

**의미:**
- 클릭뉴스와 후보뉴스 간의 "인기를 결정하는 컨텐츠"가 얼마나 유사한가?

**계산:**
```
popularity_matching = (1/N) * Σ(p_j · p_c)
```
- N: 클릭뉴스 개수
- p_j: 각 클릭뉴스의 popularity-aware representation

**해석:**
- 높을수록 **popularity bias**로 인한 클릭 가능성이 높음

---

#### Contrastive Interest Matching Score

**공식:**
```
score = σ(α · (interest_matching - β · popularity_matching))
      = σ(α · (u·f_c - β·mean(p_j·p_c)))
```

**Parameters:**
- **α:** scaling factor (default: **0.1**)
  - 범위: α ∈ (0, 1)
  - 역할: interest와 popularity matching의 전체 scale 조정
  - 작을수록: 두 matching score 차이를 완만하게 반영

- **β:** popularity penalty factor (default: **0.8**)
  - 범위: β ∈ (0, 1)
  - 역할: popularity matching의 패널티 정도 조절
  - 클수록: popularity bias를 더 강하게 억제

- **σ(x):** sigmoid function = 1 / (1 + exp(-x))

**해석:**
- **score > 0.5:** 클릭에 유저의 **interest가 더 중요하게 작용** → 추천
- **score < 0.5:** 클릭에 **popularity가 더 중요하게 작용** → 추천 억제

**목표:**
- 인기로 인해 클릭하는 게 아니라, 유저의 순수 관심사와 더 잘 매칭할수록 추천
- 이 클릭이 상대적으로 **interest-driven**인지 **popularity-driven**인지 판별

---

#### Model Training

**Click Prediction Loss: L_click**
```
L_click = -log(score)  (for positive samples)
        + -log(1 - score)  (for negative samples)
```
- Contrastive matching score를 활용한 클릭 예측 손실

**Total Loss:**
```
L_total = L_click + λ·L_pop
```

**where:**
- L_click: Contrastive interest matching 기반 클릭 예측 손실
- L_pop: Popularity disentangling 손실 (L_r + L_p + L_a)
- λ: balancing hyperparameter

---

### 5.6 전체 아키텍처 다이어그램

**Flow:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         POPCORN Framework                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  (I1) Popularity-disentangled News Modeling                            │
│  ┌──────────────┐                                                       │
│  │ News Encoder │ → h_j                                                 │
│  └──────────────┘                                                       │
│         │                                                                │
│         ├─→ Popularity-free Decoder → f_j → Predictor → ỹ_jf          │
│         └─→ Popularity-aware Decoder → p_j → Predictor → ŷ_jp         │
│                                                                          │
│         f_j, p_j는 독립적으로 유지                                     │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  (I2) Hot-to-Warm User Modeling                                        │
│  ┌────────────────────────────────┐                                     │
│  │ Candidate-guided News Selection│                                     │
│  └────────────────────────────────┘                                     │
│         │                                                                │
│         ↓ (f_j 사용)                                                    │
│    r_j = g_i ⊙ (α_j · f_j) + (1 − g_i) ⊙ f_j                           │
│         │                                                                │
│         ↓                                                                │
│  ┌──────────────┐                                                       │
│  │ User Encoder │ → u                                                   │
│  └──────────────┘                                                       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  (I3) Topic-wise Popularity Modeling                                    │
│  - Topic-wise Popularity Normalization                                  │
│  - Topic-wise Popularity Prediction                                     │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  (I4) Popularity-aware Contrastive Interest Matching                   │
│                                                                          │
│  Input:                                                                  │
│    - User: u                                                             │
│    - Candidate: f_c, p_c                                                │
│    - Clicked: {p_j}                                                     │
│                                                                          │
│  Two Matchings:                                                          │
│    (1) Interest Matching: u · f_c                                       │
│    (2) Popularity Matching: mean(p_j · p_c)                             │
│                                                                          │
│  Contrastive Score:                                                      │
│    score = σ(α · (u·f_c - β·mean(p_j·p_c)))                            │
│                                                                          │
│    score > 0.5: interest-driven → 추천                                  │
│    score < 0.5: popularity-driven → 억제                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 5.7 Loss Functions 정리

**Total Loss:**
```
L_total = L_click + λ·L_pop
```

**L_click (Click Prediction Loss):**
- Contrastive interest matching score를 활용한 클릭 예측 손실
- Binary cross-entropy 또는 negative sampling 기반

**L_pop (Popularity Disentangling Loss):**
```
L_pop = L_r + L_p + L_a
```

**where:**
- `L_r = 1/2 * ||Dense([f_j ; p_j]) - h_j||²` (Reconstruction Loss)
- `L_p = -y_j log(ŷ_jp)` (Popularity Prediction Loss)
- `L_a = -1 / (y_j log(ỹ_jf))` (Adversarial Loss)

---

### 5.8 핵심 설계 원칙

1. **Representation과 Matching의 명확한 분리**
   - f_j와 p_j를 **순수하게 유지** (scaling/weighting 없음)
   - Matching 단계에서 두 signal을 **독립적으로 활용**

2. **Matching 단계에서의 Contrastive 비교**
   - Interest matching과 popularity matching을 **명시적으로 대조**
   - 상대적 중요도를 통해 추천 결정

3. **정보 보존 및 해석 가능성**
   - f_j: interest의 순수한 clue 보존
   - p_j: popularity의 순수한 clue 보존
   - score > 0.5 / < 0.5로 **명확한 판단 기준** 제공

---

## 6. 기존 연구 비교

### 6.1 Existing Approaches: Addressing Popular News as Bias

기존 연구들은 유저의 선호도를 표현할 때, popularity로 인한 클릭을 편향으로 봐줌 [2-5]

**왜?**
- Popular해서 클릭했다는 건, contents를 진짜 좋아해서 클릭한 것이 아니기 때문
- Popularity가 클릭을 유도한 건 맞기 때문에 클릭 예측에는 활용한 연구도 있음 [3]

#### PP-Rec (ACL'21) [2]
- 뉴스의 클릭 수 기반 popularity를 time-aware popularity로 정의
- 이를 유저의 선호도 표현과 분리해서 학습
- 인기도를 직접 feature로 활용하면서, 시간적으로 decay되는 효과를 반영해 bias를 교정

#### PENR (CIKM'21) [3]
- Popularity를 bias로 보지만 제거하지는 않고, 유저의 클릭을 유도하는 하나의 signal로 간주
- Popularity로 인한 click을 bias라 간주하지만 유저 모델링 과정에서는 제거 안함
- CTR 예측 직전에 bias를 보정하는 feature로 활용해줌

#### TCCM (CIKM'23) [4]
- 인기 있는 뉴스일수록 더 많이 노출되므로 클릭 로그에 selection bias가 발생
- Time factor(freshness)를 고려해 popularity에 의해 유발된 편향을 줄임

#### ODIN (WWW'23) [5]
- 그래프에서 high-degree 노드(많이 연결된 노드 = 인기 아이템/유저)에 더 많은 정보가 집중되는 편향이 존재
- Degree-related bias term을 명시적으로 분리하고, genuine interest representation과 disentangle해서 유저의 선호도를 표현

#### **Popcorn (Ours)**
- **Popularity click 중 일부는 유저의 선호도 signal일 수 있기 때문에** 유저를 표현하는 과정에서 정교하게 반영해주어야 함
- **Representation 단계:** f_j와 p_j를 독립적으로 분리
- **Matching 단계:** Interest와 popularity를 contrastive하게 비교하여 추천 결정

---

### 6.2 POPCORN의 주요 차별점

| 측면 | 기존 연구 | POPCORN |
|-----|---------|---------|
| **Hot news 처리** | 모두 bias로 간주 또는 모두 무시 | Interest signal과 bias로 구분하여 독립적 representation 유지 |
| **Hot user 고려** | Cold/Warm user만 구분 | Hot user를 별도로 정의하고 candidate-guided selection 적용 |
| **토픽별 인기도** | 전체 뉴스에 동일한 CTR 기준 | 토픽별 상대적 인기도를 정규화하여 반영 |
| **Bias 처리 시점** | CTR 예측 직전 또는 feature로만 활용 | Representation 분리 + Matching 단계에서 contrastive 비교 |
| **Matching 방식** | 단일 score (u · c) | Dual matching (interest vs popularity) + contrastive scoring |
| **모델 통합성** | 특정 모델에 종속적 | Model-agnostic plug-in 설계 |

---

## 7. 참고문헌

[1] Wu, Chuhan, et al. "Personalized news recommendation: Methods and challenges." *ACM Transactions on Information Systems* 41.1 (2023): 1-50. (citation: 233)

[2] Wu, Chuhan, et al. "PP-Rec: News Recommendation with Personalized User Interest and Time-aware News Popularity." *In Proceedings of the ACL 2021.*

[3] Liu, Zihan, et al. "PENR: Popularity-Enhanced News Recommendation with Multi-View Interest Representation." *In Proceedings of the CIKM 2021.*

[4] Chen, Yewang, et al. "TCCM: Time and Content-Aware Causal Model for Unbiased News Recommendation." *In Proceedings of the CIKM 2023.*

[5] Hyunsik Yoo, et al. "Disentangling Degree-related Biases and Interest for Out-of-Distribution Generalized Directed Network Embedding." *In Proceedings of the WWW 2023.*

[6] Manal A. Alshehri, et al. "Generative Adversarial Zero-Shot Learning for Cold-Start News Recommendation." *In Proceedings of the ACM CIKM 2022.*

[7] Hao Jiang, et al. "Self-supervised Contrastive Enhancement with Symmetric Few-shot Learning Towers for Cold-start News Recommendation." *In Proceedings of the ACM CIKM 2023.*

[8] Wu, Chuhan, et al. "Personalized news recommendation: Methods and challenges." *ACM Transactions on Information Systems* 41.1 (2023): 1-50.

[9] Agrawal, Abhijnan, et al. "The Cloak of Anonymity: Characteristics and Linguistic Patterns in Online News Comments across Popularity and Anonymity Levels." *In Proceedings of the ACM WWW 2017.*

[10] Potthast, Martin, et al. "Clickbait Detection." *In Proceedings of the ACM SIGIR 2016.*

[11] Gulla, Jon Atle, et al. "The Adressa Dataset for News Recommendation." *In Proceedings of the ACM RecSys 2017.*

[12] De Francisci Morales, Gianmarco, et al. "Auditing News Curation Systems: A Case Study Examining Algorithmic and Editorial Logic in Apple News." *In Proceedings of the AAAI ICWSM 2021.*

---

## 8. TODO 및 추가 작업 항목

### 8.1 실험 데이터 완성
- [ ] P3 실험 데이터 입력 (토픽별 평균 CTR)
- [ ] Topic-wise lifetime visualization 데이터 정리

### 8.2 구현 세부사항 결정
- [ ] Popularity Predictor: Naive ver. vs Enhanced ver. 선택
- [ ] Contrastive matching의 α, β 값 결정 (초기값 제안 필요)
- [ ] Loss weight λ 결정
- [ ] Candidate-guided selector의 K 값 결정 방법
- [ ] ε 값 범위 실험 계획

### 8.3 Base Model 선택
- [ ] Plug-in할 News Encoder 선택 (e.g., NRMS, NAML, PLM-NR 등)
- [ ] Plug-in할 User Encoder 선택

### 8.4 데이터셋 및 실험 설계
- [ ] 실험 데이터셋 선택 (Adressa, MIND, 기타)
- [ ] Evaluation metrics 정의
- [ ] Baseline 모델 선정

### 8.5 추가 문서화
- [ ] 모듈별 상세 input/output shape 명세
- [ ] Hyperparameter 리스트 및 초기값
- [ ] 학습 파이프라인 상세 설명
- [ ] I4 Contrastive Matching의 구현 예시 코드

---

## 9. 수정 이력

| 날짜 | 버전 | 수정 내용 | 작성자 |
|-----|------|----------|--------|
| 2026-01-21 | 1.0 | 초안 작성 | - |
| 2026-01-21 | 2.0 | I4 Contrastive Interest Matching 추가, f_j/p_j 독립 유지 설계 확정 | - |
| 2026-01-21 | 2.1 | 11가지 추가 요구사항 반영: Title input 명시, Decoder 원리 상세화, Topic embedding 구체화, Popularity Predictor 원리 설명, L_pop 적용 시점 명시, Target-aware Attention 설계, Gated Residual Connection 상세 설명, gi 메커니즘 분석, Topic-wise normalization PENR 차이점 명확화, α=0.1/β=0.8 default 값 추가, 버전 표기 정리 | - |
| 2026-01-22 | 2.2 | Component (I2) 및 (I3) 구현 상세화: (1) W_g 학습 메커니즘 완전 설명 (Computational Graph, Gradient Flow, L_click 연결, 학습 목표 및 패턴 예시, PyTorch 구현), (2) r_j 도출 전체 파이프라인 구현 (CandidateGuidedNewsSelector 클래스, 4단계 프로세스, 데이터 흐름, 사용 예시 2개), (3) PENR 기반 인기도 계산 상세 설명 (변수 의미, 시간 윈도우, 정규화 3단계, 구체적 계산 예시, PENR 한계 분석), (4) POPCORN Topic-wise Re-normalization 완전 구현 (compute_topic_wise_popularity 함수, popularity_to_class 10-class binning, PENR vs POPCORN 비교 테이블) | - |

---

**문서 상태:** 구현 명세 상세화 완료
