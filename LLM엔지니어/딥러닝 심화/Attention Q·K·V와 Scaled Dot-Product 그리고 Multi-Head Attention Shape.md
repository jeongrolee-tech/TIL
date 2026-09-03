# 2026-09-03 Attention Q·K·V와 Scaled Dot-Product 그리고 Multi-Head Attention Shape

## 오늘의 TIL 요약

**3장1강. Attention의 필요성과 Query/Key/Value**
- 같은 토큰이라도 **주변 문맥에 따라 참고해야 할 대상이 달라짐**
  - "강아지가 공을 물었다. **그것**은 신났다" → 그것 ≈ 강아지
  - "강아지가 공을 물었다. **그것**은 빨갛다" → 그것 ≈ 공
  - 토큰 ID는 동일한데 필요한 의미가 다르므로, **고정된 embedding 하나로는 부족**
- **Self-Attention이 하는 일** — 표현의 업데이트
  ```
  기본 토큰 표현  +  주변 토큰과의 관계  =  문맥이 반영된 새 토큰 표현
                                            (contextual representation)
  ```
  - ⭐ 토큰을 **새로운 ID로 교체하는 과정이 아님** — 같은 위치의 벡터를 문맥에 맞게 다시 쓰는 과정
  - 모든 토큰을 같은 비율로 섞지 않고, **관련 있는 위치에 더 큰 비율**을 줌
- **Query / Key / Value의 역할** — 검색 비유로 나누면 헷갈리지 않음

  | 역할 | 질문 | 검색 비유 |
  | --- | --- | --- |
  | Query | 지금 이 토큰은 무엇을 알고 싶은가? | 검색어 |
  | Key | 이 후보는 그 질문과 얼마나 관련 있는가? | 색인·태그 |
  | Value | 이 후보에서 실제로 가져올 내용은 무엇인가? | 검색 결과 본문 |

  - 💡 **Weight는 Key와의 비교로 만들지만, 실제로 곱해지는 대상은 Value** — "무엇으로 찾을지"와 "무엇을 가져올지"를 분리한 설계
- **Projection = 역할별 선형 변환**
  - 딥러닝 기초의 `Linear` layer와 같음: `x → Linear → 새로운 표현`
  - Self-Attention에서는 **같은 입력 X를 세 관점으로 다시 표현**
    ```python
    import torch
    from torch import nn

    D = 8
    x = torch.randn(2, 5, D)            # [B, L, D]

    to_q = nn.Linear(D, D, bias=False)  # 세 층 모두 D→D지만
    to_k = nn.Linear(D, D, bias=False)  # 학습되는 가중치는 서로 다름
    to_v = nn.Linear(D, D, bias=False)

    q, k, v = to_q(x), to_k(x), to_v(x)
    # Linear는 마지막 축 D만 변환 → B, L은 그대로 유지
    assert q.shape == k.shape == v.shape == (2, 5, D)
    ```
- **자주 생기는 오해 4가지**
  - ⛔ "Q, K, V는 서로 다른 세 문장이다" → Self-Attention에서는 **같은 입력 시퀀스**에서 출발
  - ⛔ "Query는 실제 자연어 질문이다" → Query는 **학습된 벡터**이고, "무엇을 찾는가"는 비유일 뿐
  - ⛔ "Attention은 가장 관련 있는 토큰 하나만 고른다" → 일반적인 soft attention은 **여러 토큰을 비율대로 함께 섞음**
  - ⛔ "높은 weight = 인간이 생각하는 중요도" → 모델 내부의 참고 패턴 단서일 뿐, 완전한 설명이 아님

- 📝 **이해도 점검**
  1. Self-Attention이 필요한 이유를 한 문장으로?
     → 현재 토큰이 **다른 토큰과의 관계를 참고해 문맥에 맞는 표현**을 만들기 위해
  2. Q, K, V 중 실제로 정보가 흘러나오는 것은?
     → **Value** (Q·K는 섞을 비율을 정하는 데만 쓰임)

**3장2강. Scaled Dot-Product Attention 계산**
- **공식과 네 단계**
  ```
  Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
  ```

  | 단계 | 연산 | 의미 |
  | --- | --- | --- |
  | ① 비교 | `Q @ Kᵀ` | Query와 Key의 관련도 score |
  | ② 조절 | `/ √d_k` | score 크기 완화 (scaling) |
  | ③ 비율 | `softmax(dim=-1)` | 합이 1인 attention weight |
  | ④ 섞기 | `weights @ V` | Value 가중합 → 새 표현 |

- **Score matrix는 토큰×토큰 관계표**
  - **행(row)** = 의미를 업데이트하려는 **Query**
  - **열(column)** = 참고 후보인 **Key**
  - **칸(cell)** = 그 Query–Key 쌍의 관련도 점수
  - ⛔ Score는 확률이 아님 — **음수도 가능하고 합이 1일 필요도 없음** (예: `[-0.5, 1.7, 0.2]`)
  - Softmax를 통과해야 비로소 "참고 비율"로 읽을 수 있는 weight가 됨
- **Scaling을 하는 이유**
  - 차원 `d_k`가 커질수록 dot-product는 **더 많은 항이 더해져 값의 분산이 커짐**
  - 큰 값이 Softmax에 들어가면 분포가 지나치게 **뾰족(peaky)** 해져 한 위치에 weight가 몰리고, gradient가 거의 0인 구간으로 밀림
  - ⭐ `√d_k`로 나누는 것은 **순위를 바꾸지 않고 크기만 완화**해 학습을 안정화하는 장치
    ```python
    scores = q @ k.transpose(-2, -1) / (D ** 0.5)   # [B, L, L]
    weights = torch.softmax(scores, dim=-1)         # 각 행의 합 = 1
    context = weights @ v                           # [B, L, D]
    ```
- **왜 `dim=-1`인가**
  - score shape이 `[B, L_query, L_key]`이므로 **마지막 축이 Key 후보 축**
  - 한 Query가 모든 Key에 나눠 주는 비율을 만들어야 하므로 마지막 축에 Softmax
  - `weights.sum(dim=-1)`은 Query마다 약 `1.0`
- **Shape 흐름** (Q, K, V가 모두 `[B, L, D]`일 때)

  | 단계 | Tensor | Shape | 의미 |
  | --- | --- | --- | --- |
  | 입력 | Q | `[B, L, D]` | 각 토큰의 질문 표현 |
  | 입력 | K | `[B, L, D]` | 각 토큰의 비교 기준 |
  | 입력 | V | `[B, L, D]` | 각 토큰이 내줄 정보 |
  | 비교 | `Q @ Kᵀ` | `[B, L, L]` | 토큰×토큰 관련도표 |
  | 비율 | `softmax` | `[B, L, L]` | 토큰×토큰 참고 비율 |
  | 혼합 | `weights @ V` | `[B, L, D]` | 문맥 반영 토큰 표현 |

  - ⭐ 핵심 변화: **`[B,L,D]` → 관계표 `[B,L,L]` → 다시 `[B,L,D]`** (입출력 shape이 같아 layer를 쌓을 수 있음)
- **자주 나는 오류**
  - ⛔ `K.transpose(-2, -1)` 누락 → `[B,L,D] @ [B,L,D]`는 원하는 행렬곱이 아님. K를 `[B,D,L]`로 바꿔야 토큰×토큰이 나옴
  - ⛔ `softmax(dim=1)` 사용 → Query 축에 정규화가 걸려 "각 Key가 받은 비율"이라는 엉뚱한 값이 됨
  - ⛔ score를 그대로 확률로 해석 → 해석은 **Softmax 이후** 값으로
  - ⛔ scaling이 shape을 바꾼다고 생각 → **숫자 크기만 조절**, shape은 그대로

- 📝 **이해도 점검**
  1. 계산 네 단계 순서는? → **Q/K 비교 → scaling → softmax → Value 가중합**
  2. Score matrix의 행과 열은? → 행은 **Query**, 열은 **Key**
  3. Score와 weight의 차이는? → score는 **정규화 전 관련도**, weight는 **Softmax 후 합이 1인 참고 비율**
  4. Softmax를 마지막 축에 적용하는 이유는? → 마지막 축이 **한 Query가 비교하는 Key 후보 축**이므로
  5. `[B,L,D]`일 때 score와 output shape은? → ⭐ score `[B,L,L]`, output `[B,L,D]`

**3장3강. Attention weight와 Context Vector 해석**
- **Attention weight = 한 Query가 각 Key를 참고하는 비율**
  ```
  "먹었다" Query 행
    민수는    0.25
    식당에서  0.15
    라면을    0.45   ← 가장 많이 참고
    먹었다    0.15   ← 자기 자신도 후보
    ─────────────
    합계      1.00
  ```
  - 가장 큰 값이 있어도 **나머지 정보를 버리는 것이 아님** — 작은 비율로 함께 섞임
  - ⛔ weight는 **정답 확률이 아님** — 특정 layer, 특정 head에서 정보를 섞는 비율일 뿐
- **Heatmap 읽는 법** — 행렬이므로 색으로 그리면 heatmap
  - **한 행씩** 읽는다: "이 Query가 어떤 Key를 많이 봤는가"
  - 행 = Query, 열 = Key, 한 행의 합 = 1
  - 💡 열 방향 합계는 정규화된 값이 아니므로 "이 토큰이 얼마나 중요한가"로 곧바로 읽으면 안 됨
- **Context vector = Value의 가중합**
  ```
  context[i] = Σ_j  weight[i, j] × V[j]
  ```
  - 특정 단어 하나를 **복사한 것이 아니라**, 여러 위치의 정보가 비율대로 섞인 **새로운 벡터**
  - Weight는 **Key와 비교해 만들지만 마지막엔 Value에 곱해짐** — 헷갈리기 쉬운 지점
- **해석할 때의 한계**
  - Attention weight는 **관찰 도구**이지 최종 예측의 완전한 설명이 아님
  - 실제 출력에는 weight 외에 **Value·W_O·FFN·residual·여러 layer의 누적**이 함께 작용
  - ⛔ "낮은 weight = 영향 없음", "높은 weight = 근거" 같은 단정은 위험

- 📝 **이해도 점검**
  1. weight 한 행이 뜻하는 것은? → **Query 하나가 모든 Key를 참고하는 비율**
  2. context vector는 어떻게 만들어지나? → weight를 **같은 위치의 Value**에 곱해 모두 더함
  3. heatmap의 행/열은? → 행 Query, 열 Key
  4. context vector와 Value 하나의 차이는? → context는 **여러 Value가 weight에 따라 섞인 새 표현**
  5. weight를 완전한 설명으로 보면 안 되는 이유는? → **특정 layer/head의 참고 패턴**일 뿐, 예측에 관여하는 모든 계산을 단독으로 설명하지 못하므로

**3장4강. Tensor로 Self-Attention 미니 구현**
- **먼저 shape 계약을 적는다** — 함수 내부 계산이 맞는지 검증하기 쉬워짐

  | 항목 | Shape | 의미 |
  | --- | --- | --- |
  | 입력 `x` | `[B, L, D]` | 토큰의 현재 표현 |
  | 선택 입력 `valid_key_mask` | `[B, L]` | 실제 토큰인지 여부(PAD 구분) |
  | 출력 `context` | `[B, L, D]` | 문맥이 반영된 새 표현 |
  | 출력 `weights` | `[B, L, L]` | Query별 Key 참고 비율 |

- **기호 정리**

  | 기호 | 의미 | 예시 |
  | --- | --- | --- |
  | `B` (Batch Size) | 한 번에 처리하는 문장 수 | 32 |
  | `L` (Sequence Length) | 한 문장의 토큰 수 | 128 |
  | `D` (Hidden / Embedding Dim) | 토큰 하나를 나타내는 벡터 길이 | 768 |

- **교육용 단순화: `Q = K = V = x`**
  - projection을 잠시 생략하면 "**모든 토큰이 서로 비교되고, weight로 정보를 섞는다**"는 흐름만 남음
  - ⛔ 실제 모델은 이렇게 하지 않음 — 학습 가능한 `W_Q, W_K, W_V`가 있어야 역할별 표현이 분리됨
  ```python
  import torch

  def toy_self_attention(x, valid_key_mask=None):
      # x: [B, L, D]
      B, L, D = x.shape

      scores = x @ x.transpose(-2, -1)          # [B, L, L]  토큰×토큰
      scores = scores / (D ** 0.5)              # scaling (shape 불변)

      if valid_key_mask is not None:            # [B, L] → [B, 1, L]로 broadcast
          # ⭐ mask는 반드시 Softmax "전"에 적용
          scores = scores.masked_fill(~valid_key_mask[:, None, :], float("-inf"))

      weights = torch.softmax(scores, dim=-1)   # [B, L, L]  Key 축 정규화
      context = weights @ x                     # [B, L, D]
      return context, weights

  x = torch.randn(2, 5, 8)
  mask = torch.tensor([[1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 0]], dtype=torch.bool)
  context, weights = toy_self_attention(x, mask)

  assert context.shape == (2, 5, 8) and weights.shape == (2, 5, 5)
  assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
  ```
- **Masking의 위치가 중요한 이유**
  - Softmax **이후**에 0을 곱하면 남은 값들의 합이 1이 아니게 됨
  - Softmax **이전**에 `-inf`(또는 아주 작은 값)로 만들면 `exp(-inf) = 0`이 되어 **비율이 자동으로 재분배**됨
  - 💡 Decoder의 causal mask도 같은 원리 — 미래 위치의 score를 Softmax 전에 막음
  - ⛔ 한 행 전체가 마스킹되면 `NaN`이 발생 — PAD 행 처리를 별도로 확인
- **디버깅 순서**: ① shape → ② `transpose` 여부 → ③ Softmax 축 → ④ mask 적용 시점

- 📝 **이해도 점검**
  1. 이번 구현에서 Q, K, V는? → **`Q = K = V = x`로 단순화**
  2. `x [B,L,D] @ xᵀ [B,D,L]`의 결과는? → `[B, L, L]`
  3. Softmax 축은? → Key 후보가 있는 **마지막 축 `dim=-1`**
  4. PAD Key의 weight를 0으로 만들려면 언제 mask? → **Softmax 전에** 매우 작은 값 또는 `-inf`로 치환
  5. output shape이 입력과 같은 이유는? → Query 위치마다 **Value 차원 `D`의 새 표현 하나**를 만들기 때문

**3장5강. Multi-Head Attention Shape과 디버깅**
- 공식을 손으로 여러 번 계산하는 것이 목표가 아니라, **"`D` 차원을 여러 head가 나눠 계산하고 다시 합친다"는 shape 흐름**을 잡는 것이 목표
- **왜 여러 head인가** — 같은 입력을 **서로 다른 projection으로 바라볼 기회**를 줌
  - 어떤 head는 주어–동사 관계를, 다른 head는 수식어–명사 관계를, 또 다른 head는 멀리 떨어진 위치 관계를 학습할 수 있음
  - ⛔ 각 head의 역할을 사람이 지정하는 것이 아니라 **학습 결과로 나뉘는 경향**일 뿐
- **차원 분할 규칙**
  ```
  D_head = D_model / num_heads       (D_model % num_heads == 0 이어야 함)

  D_model = 12,  num_heads = 3
    Head1: 4     Head2: 4     Head3: 4     ← 각각 D_head
  ```
  - `D_model` = 모델(Model)이 사용하는 **전체 벡터 차원(Dimension)**, `D_head` = **각 head가 맡는 몫**
  - 💡 head를 늘려도 전체 계산량은 대체로 비슷 — **같은 예산을 쪼개 쓰는 구조**
- **Q, K, V는 입력을 세 번 복사한 값이 아님**
  ```
  X: [B, L, D_model]
  Q = X·W_Q,  K = X·W_K,  V = X·W_V     → 각각 [B, L, D_model]
  ```
  - `W_Q, W_K, W_V`는 사람이 미리 정한 규칙이 아니라 **역전파로 갱신되는 파라미터**
  - ⭐ 정확한 표현은 "입력을 똑같이 세 번 복사한다"가 아니라 "**같은 입력을 서로 다른 관점으로 선형변환한다**"
- **Head 분리는 선형 투영 "다음"에 일어남**
  ```
  X             [B, L, D_model]
   → Q/K/V 투영  [B, L, D_model]
   → head 분리   [B, L, H, D_head] → 축 이동 → [B, H, L, D_head]
   → head별 Attention 계산
   → head 결과   [B, H, L, D_head]
   → 결합        [B, L, H, D_head] → [B, L, D_model]
   → W_O 투영    [B, L, D_model]
  ```
  - head 축을 앞으로 옮기는 이유는 **B개 샘플 × H개 head를 한 번에 병렬 계산**하기 위함
- **Concat 뒤의 `W_O`가 필요한 이유**
  - 단순히 이어 붙이면 각 head의 결과가 **옆에 나란히 배치된 상태**일 뿐
  - 학습 가능한 `W_O`가 **head 간 정보를 다시 섞어** 다음 sublayer가 쓸 표현으로 바꿔줌
  - ⛔ `W_O`까지 포함해야 Multi-Head Attention의 출력 투영이 완성됨
- **PyTorch `nn.MultiheadAttention` (`batch_first=True`)**

  | 항목 | Shape |
  | --- | --- |
  | Query 입력 | `[B, L_query, D_model]` |
  | Key / Value 입력 | `[B, L_key, D_model]` |
  | Output | `[B, L_query, D_model]` |
  | head별 weight | `[B, H, L_query, L_key]` |

  ```python
  mha = torch.nn.MultiheadAttention(embed_dim=12, num_heads=3, batch_first=True)
  x = torch.randn(2, 5, 12)

  out, w_avg = mha(x, x, x)                                # w_avg: [2, 5, 5]
  out, w_all = mha(x, x, x, average_attn_weights=False)    # w_all: [2, 3, 5, 5]
  ```
  - ⛔ `average_attn_weights`의 기본값은 `True` → **head 축이 평균으로 사라짐**. head별 패턴을 보려면 `False`
  - ⛔ `batch_first`를 빠뜨리면 축 순서가 `[L, B, D]`로 해석되어 조용히 잘못된 결과가 나옴
  - 디버깅 시 확인 순서: `D_model` → `num_heads` → `batch_first` → `average_attn_weights`
- **선택 학습 · MHA / MQA / GQA**

  | 구조 | Query head | Key/Value head | 특징 |
  | --- | --- | --- | --- |
  | MHA | `H_q` | `H_kv = H_q` | 각 Query head가 전용 K/V를 가짐 |
  | MQA | `H_q` | `H_kv = 1` | 모든 Query head가 하나의 K/V를 공유 |
  | GQA | `H_q` | `1 < H_kv < H_q` | Query head **그룹**이 K/V를 공유 |

  - 예: `Q [B,32,L,D_head]` + `K/V [B,8,L,D_head]` → 4개 Query head가 한 K/V head를 공유
  - ⭐ **KV Cache 크기는 `H_q`가 아니라 `H_kv`에 비례** → GQA는 긴 문맥 추론의 메모리를 줄이는 데 유리
  - 💡 그래서 Decoder-only LLM config에는 Query head 수와 K/V head 수가 따로 적혀 있을 수 있음

- 📝 **이해도 점검**
  1. Multi-Head Attention을 한 문장으로? → **같은 입력을 여러 head가 서로 다른 관점으로 Attention한 뒤 결과를 합치는 구조**
  2. `D_model=12, num_heads=3`일 때 `D_head`는? → **4**
  3. head로 나눈 뒤 대표 shape은? → `[B, H, L, D_head]`
  4. head별 weight를 보려면? → `average_attn_weights=False`
  5. `D_model % num_heads == 0`이 필요한 이유는? → 각 head가 **동일한 정수 크기의 `D_head`**를 가져야 하므로

**오늘 배운 것 한눈에 정리**
- Self-Attention은 토큰을 새 ID로 바꾸는 것이 아니라, **주변 토큰과의 관계를 반영해 같은 위치의 표현을 업데이트**하는 과정
- Q는 "무엇을 찾는가", K는 "비교 기준", V는 "실제로 가져올 정보" — **모두 같은 입력 X에서 서로 다른 학습 가능한 선형변환**으로 만들어짐 (복사가 아님)
- 계산은 **비교(`QKᵀ`) → 조절(`/√d_k`) → 비율(`softmax(dim=-1)`) → 섞기(`@V`)** 의 네 단계
- Scaling은 **순위를 바꾸지 않고 값의 크기만 완화**해 Softmax가 지나치게 뾰족해지는 것을 막음
- ⭐ Shape의 핵심 흐름: **`[B,L,D]` → 관계표 `[B,L,L]` → `[B,L,D]`** — 입출력 shape이 같아 layer를 쌓을 수 있음
- Score는 정규화 전 값이라 **확률이 아니고**, Softmax 이후의 weight만 "참고 비율"로 읽을 수 있음
- Context vector는 **Value의 가중합** — 특정 단어의 복사본이 아니라 여러 위치가 비율대로 섞인 새 벡터
- Attention weight(heatmap)는 **행이 Query, 열이 Key**이며 유용한 관찰 도구지만 **예측의 완전한 설명은 아님**
- Padding·causal mask는 반드시 **Softmax 전에** 적용해야 남은 위치들의 비율이 올바르게 재분배됨
- Multi-Head는 `D_head = D_model / num_heads`로 나누고 **투영 → head 분리 → head별 Attention → 결합 → `W_O`** 순서이며, `W_O`까지 있어야 출력 투영이 완성됨
- MQA·GQA는 **K/V head 수를 줄여 KV Cache 메모리를 절약**하는 변형 (`H_kv`에 비례)

**데일리 퀴즈 정리**
1. Self-Attention의 Query·Key·Value → **같은 입력 토큰 표현에서 서로 다른 학습 가능한 선형변환을 통해 만들어짐**
   - `self`가 뜻하는 것은 Q, K, V가 **같은 입력 시퀀스에서 출발**한다는 점이며, 서로 다른 projection을 지나므로 값은 달라짐
   - ⛔ Query는 실제 문장이 아니라 **학습되는 벡터**
   - Key는 **비교 기준**, Value는 **실제로 가져올 정보** — 이 둘의 역할을 뒤집어 서술한 설명은 틀림
2. `softmax(QKᵀ/√d_k)V`에서 Softmax 전에 `√d_k`로 나누는 이유 → **차원이 커지며 함께 커진 score를 완화해 weight가 한 위치로 쏠리는 것을 막기 위해**
   - `d_k`가 커지면 dot-product score도 커지기 쉽고, 큰 값이 들어간 Softmax는 한 위치에 weight가 몰림
   - ⭐ scaling은 **순위를 바꾸는 것이 아니라 값의 크기만 완화**해 학습을 안정화하는 역할
3. Q, K, V가 모두 `[B, L, D]`일 때 `Q @ Kᵀ`와 최종 context의 shape → **`Q@Kᵀ`: `[B,L,L]`, context: `[B,L,D]`**
   - K를 transpose해 `[B,D,L]`로 만든 뒤 `Q [B,L,D]`와 곱하면 토큰×토큰 관계표 `[B,L,L]`
   - 이어서 `weights [B,L,L] @ V [B,L,D]` → **입력과 같은 `[B,L,D]`**
4. Attention weight(heatmap)에 대한 옳은 설명 → **특정 layer/head의 참고 패턴을 보여주는 관찰 도구이며 완전한 예측 설명은 아님**
   - ⛔ "높은 weight가 항상 인간이 생각하는 중요도와 일치한다", "낮은 weight는 영향이 전혀 없다"는 모두 오해로 소개됨
   - heatmap의 **행은 Query, 열은 Key**
5. `D_model=12, num_heads=3`일 때 `D_head` → **4**
   - `D_head = D_model / num_heads`
   - 각 head가 **동일한 정수 크기**를 가져야 하므로 `D_model`이 `num_heads`로 나누어떨어져야 함

**한 줄 정리**: Self-Attention은 같은 입력 X를 서로 다른 학습 가능한 투영으로 Q(무엇을 찾는가)·K(비교 기준)·V(가져올 정보)로 다시 표현한 뒤 `QKᵀ`로 토큰×토큰 관계표를 만들고 `√d_k`로 크기를 완화해 마지막 Key 축에 Softmax를 적용해 합이 1인 참고 비율을 얻고 그 비율로 Value를 가중합하는 **`[B,L,D] → [B,L,L] → [B,L,D]`의 표현 업데이트**이며 — 이때 score는 확률이 아니고 context vector는 특정 단어의 복사본이 아니며 attention weight는 행이 Query·열이 Key인 유용한 관찰 도구일 뿐 예측의 완전한 설명은 아니고, PAD·causal mask는 반드시 Softmax 전에 적용해야 비율이 올바르게 재분배되며, Multi-Head는 `D_head = D_model / num_heads`로 차원을 쪼개 `[B,H,L,D_head]`로 병렬 계산한 뒤 다시 `[B,L,D_model]`로 합치고 `W_O`로 head 간 정보를 섞어야 출력 투영이 완성되고, K/V head 수를 줄인 MQA·GQA는 `H_kv`에 비례하는 KV Cache를 줄여 긴 문맥 추론에 유리하다.
