# 2026-08-21 활성화 함수와 출력층 설계 (ReLU / Sigmoid / Softmax)

## 오늘의 TIL 요약

**4장1강. 비선형성과 활성화 함수 필요성**
- 선형층(`nn.Linear`)만 연속으로 쌓아도 전체 계산은 결국 하나의 아핀 변환 `y = ax + b`로 합쳐짐
  - `h = x @ W1.T + b1`, `out = h @ W2.T + b2`를 정리하면 `W_new = W2 @ W1`, `b_new = b1 @ W2.T + b2`인 `out = x @ W_new.T + b_new` 하나로 압축됨
  - 층 수가 많거나 중간 feature 차원이 커도(예: 100개) 동일 — "층이 많다"가 자동으로 "복잡한 함수를 표현한다"를 뜻하지 않음
- ReLU처럼 구간에 따라 동작이 달라지는 함수가 들어가면 더 이상 하나의 직선으로 합쳐지지 않음 → 활성화 함수가 비선형성을 만들어주는 지점
- 활성화 함수는 Tensor의 **shape를 바꾸지 않고 값만 비선형적으로 변형**함
- 보통 은닉층 뒤에 넣으며, 은닉층의 활성화 함수(ReLU 등)와 출력층의 후처리(Sigmoid/Softmax)는 서로 다른 역할이므로 섞어서 생각하면 안 됨
- `nn.Identity()`(입력을 그대로 통과)와 비교하면 "활성화 함수 없음" 상태를 실험적으로 재현 가능
- **학습 실패(최적화 문제)와 구조적 표현 불가(표현력 문제)는 다름**
  - ReLU 모델: XOR 같은 패턴을 표현할 수 있지만 seed/learning rate가 나쁘면 한 번의 실행에서 못 찾을 수 있음
  - Linear-only 모델: 아무리 오래 학습해도 구조적으로 XOR을 표현 불가능

**4장2강. ReLU의 역할과 사용 위치**
- `ReLU(x) = max(0, x)` — 음수는 0, 양수는 그대로 통과
- 계산이 단순하고 양수 구간에서는 gradient가 1로 잘 흐름, 입력/출력 shape는 동일
- 은닉층의 Linear 뒤에 넣는 것이 일반적이며, **출력층 뒤에 무조건 붙이면 안 됨** (이진/다중 분류/회귀에 따라 출력층 설계가 달라짐)
- Dead ReLU: 어떤 뉴런이 계속 음수만 출력하면 ReLU 이후 항상 0이 되어 gradient가 흐르지 않아 학습 신호를 거의 못 받는 상태 — 초반엔 "ReLU는 음수 구간에서 gradient가 0이 될 수 있다" 정도로만 기억

**4장3강. Sigmoid와 이진 분류 출력층**
- 이진 분류는 보통 샘플당 logit 1개 출력 → shape `(batch_size, 1)`
- 예측 흐름: `입력 → MLP → logits → sigmoid(logits) → probability → threshold → label`
- `logit < 0`은 class 0, `logit > 0`은 class 1 쪽 근거 — 확률 threshold 0.5와 logit threshold 0은 동일한 경계(`sigmoid`가 단조증가이므로). threshold를 0.5가 아닌 값으로 바꾸면 logit 기준은 `log(p/(1-p))`로 변환해야 함
- **`nn.BCEWithLogitsLoss`는 raw logits를 입력으로 받음** — 모델 마지막에 `nn.Sigmoid()`를 넣지 않고, Sigmoid는 예측 해석 단계(`torch.sigmoid(logits)`)에서만 사용
  - 이유: 매우 큰 양/음수 logit에서 확률을 먼저 계산한 뒤 log를 취하면 0으로 잘리는 등 수치적으로 불안정해질 수 있어, Sigmoid+BCE를 내부에서 안정적으로 결합 계산함
- 체크리스트: logits/target shape `(batch_size, 1)`, target dtype `float32`, loss `BCEWithLogitsLoss()`, 예측 확률 `sigmoid(logits)`, 예측 label `(probs >= 0.5).long()`
- (심화) 다중 레이블 분류는 class마다 logit을 출력(`(batch_size, num_labels)`)하고 각각에 BCEWithLogitsLoss 적용. Sigmoid 출력이 항상 잘 보정된 확률은 아니므로 중요한 업무면 validation에서 calibration 확인 필요

**4장4강. Softmax와 다중 분류 출력층**
- 다중 분류는 class 수만큼 logits 출력 → shape `(batch_size, num_classes)`
- 예측 흐름: `입력 → MLP → class별 logits → Softmax → class별 probability → argmax → 예측 class index`
- `nn.CrossEntropyLoss`는 raw logits와 정수 target을 받아 내부에서 cross entropy를 계산 → **모델 마지막에 `nn.Softmax()`를 넣지 않음**
- Softmax는 절대 점수가 아니라 **class 간 상대적 차이**로 확률을 결정 (logits를 전체적으로 같은 값만큼 올리거나 내려도 같은 분포가 나옴)
- `dim`은 어떤 축의 값들이 서로 경쟁할지 지정: `(B, C)`는 `dim=1`(또는 `-1`), `(B, L, C)`는 `dim=-1`
- target shape가 `(B,)`인 이유: 샘플마다 정답 class 번호(정수) 하나만 저장하기 때문 (dtype `long`, 값 범위 `0~C-1`)
- 예측 class만 필요하면 Softmax 없이 `argmax(logits, dim=-1)`로 충분
- 다중 분류(하나만 정답, CrossEntropyLoss) vs 다중 레이블(여러 개 동시 정답, BCEWithLogitsLoss) 구분 필요
- 체크리스트: logits `(batch_size, num_classes)`, target `(batch_size,)`, target dtype `long`, loss `CrossEntropyLoss()`, 확률 확인 `softmax(logits, dim=1)`, 예측 class `argmax(logits, dim=1)`
- 흔한 실수: target을 `(batch_size, 1)`로 만드는 것 → `target_wrong.squeeze(1)`로 수정

**4장 전체 정리**
- 활성화 함수는 MLP에 비선형성을 부여
- ReLU는 은닉층의 기본 활성화 함수
- 이진 분류: logit 1개 + `BCEWithLogitsLoss`
- 다중 분류: class 수만큼 logits + `CrossEntropyLoss`
- 출력층은 임의로 정하는 게 아니라 **문제 유형(이진/다중/다중레이블)과 loss 함수에 맞춰 설계**해야 함

**데일리퀴즈 정리**
1. `CrossEntropyLoss`와 함께 쓰는 모델은 마지막에 Softmax를 넣지 않고 raw logits를 출력해야 함 — 미리 Softmax/Sigmoid를 적용하면 확률이 이중 계산되어 loss가 왜곡됨
2. `CrossEntropyLoss`의 target dtype은 `torch.long`(class index) — float나 one-hot을 넣으면 오류/오동작
3. batch 16, class 5개인 다중 분류 모델의 logits shape는 `(16, 5)` — `(batch_size, num_classes)` 순서
4. Linear layer만 여러 개 쌓으면 결국 하나의 새로운 weight/bias를 가진 **아핀 변환 하나로 수렴**함(비선형성이 만들어지지 않음)
5. ReLU는 `max(0, x)`로 음수를 0으로, 양수는 그대로 통과시키는 함수이며 shape를 바꾸지 않음 — 0~1 확률로 바꾸는 것은 Sigmoid/Softmax의 역할이므로 혼동하지 않도록 주의(오답으로 확인한 포인트)

**한 줄 정리**: 활성화 함수는 shape를 바꾸지 않고 값에 비선형성을 부여하며(ReLU는 은닉층에), 이진 분류는 logit 1개 + Sigmoid/BCEWithLogitsLoss, 다중 분류는 class 수만큼 logits + Softmax/CrossEntropyLoss로 — 출력층은 항상 문제 유형과 loss 함수에 맞춰 설계해야 한다.
