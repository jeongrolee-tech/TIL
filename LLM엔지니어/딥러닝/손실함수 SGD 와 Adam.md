# 2026-08-24 손실 함수와 Optimizer (SGD / Adam)

## 오늘의 TIL 요약

**5장1강. 손실함수의 역할과 목표함수**
- 손실 함수(loss function): 모델 예측과 정답을 비교해 오차를 숫자 하나로 만듦 → 이 값을 줄이는 방향으로 parameter를 업데이트하는 것이 학습의 기본 흐름
- loss(샘플/배치 단위 오차) vs objective(학습에서 최종적으로 최소화할 전체 식). 예: `objective = data_loss + lambda * regularization`
- MSELoss 직접 계산: `((pred - target) ** 2).mean()` (오차 → 제곱 → 평균)
- `reduction` 옵션: `'none'`(샘플별 값 유지), `'sum'`(합계), `'mean'`(평균, 기본값) — mean은 batch size가 달라져도 loss/gradient 규모가 비교적 일관되게 유지됨
- `reduction='none'`은 그대로 `backward()` 불가 → 샘플별 오차 분석·가중치 적용·error analysis에 사용, 기본 학습 루프에서는 보통 `mean`으로 만든 scalar loss 사용
- loss는 같은 문제·같은 loss 정의 기준에서만 비교 의미가 있음 — 서로 다른 loss 함수의 절대값을 직접 비교하면 안 됨
- loss는 매 step 반드시 내려가지는 않음 — 한두 step 오를 수 있으므로 이동평균/epoch 평균/validation loss로 추세를 봐야 함
- loss vs metric(accuracy): loss는 "얼마나 확신 있게 맞았는지"까지 반영하고, accuracy는 "맞았는지/틀렸는지"만 봄 → 학습(optimizer 업데이트)에는 loss를, 모델 선택에는 validation loss + 업무 metric(Accuracy, F1, Recall 등)을 함께 사용
- 학습 한 step: `loss = criterion(...)`(채점) → `loss.backward()`(원인 분석, parameter별 gradient 계산) → `optimizer.step()`(실제 수정)

**5장2강. 회귀/이진/다중 분류 손실 선택**
- 결정 순서: 문제 정의 → 출력 shape/의미 → target shape/dtype → loss
- 회귀: 출력 실수 1개, target float → `nn.MSELoss()`
- 이진 분류: logit 1개, target `(B, 1)` float → `nn.BCEWithLogitsLoss()`
- 다중 분류(샘플당 정답 하나): class별 logits `(B, num_classes)`, target `(B,)` long, 값 범위 `0~C-1` → `nn.CrossEntropyLoss()`
- 다중 레이블(한 샘플에 여러 정답 동시 가능): label별 독립 logit `(B, num_labels)`, target 같은 shape의 0/1 float → `nn.BCEWithLogitsLoss()` (Softmax처럼 하나만 고르는 경쟁이 아니라 label별 독립적인 이진 질문)
- 양성 샘플이 매우 적은 불균형 문제는 `pos_weight`로 양성 오류 가중치를 높일 수 있음 — 먼저 데이터 분포와 metric을 확인한 뒤 적용
- 코드 작성 전 계약표 작성 습관: `output shape / target shape / target dtype / target value range / loss`를 먼저 적으면 마지막 Linear의 `out_features`나 target 전처리 오류를 크게 줄일 수 있음
- 흔한 실수: 다중 클래스(정답 하나, CrossEntropyLoss)와 다중 레이블(정답 여러 개 가능, BCEWithLogitsLoss)을 혼동하는 것

**5장3강. SGD, Adam과 Learning Rate**
- 학습 흐름: loss 계산(오차) → gradient 계산(backward, 방향) → optimizer가 parameter 업데이트(실제 수정). 공식: `새 parameter = 기존 parameter - learning_rate × gradient`
- gradient는 loss가 가장 빠르게 증가하는 방향을 가리킴 → loss를 줄이려면 그 반대 방향으로 이동
- learning rate는 단순한 "속도"가 아니라 gradient를 실제 이동량으로 바꾸는 배율 — 너무 작으면 학습이 느리거나 제한된 epoch 안에 좋은 영역에 못 도달할 수 있고, 너무 크면 흔들리거나 발산할 수 있음. 고정된 정답 공식은 없고 모델 구조·batch size·optimizer·normalization·scheduler에 따라 달라지므로 후보값을 정해 loss curve와 validation 성능으로 비교
- optimizer 비교 실험 시 데이터·초기 parameter(`state_dict()` 복사)·optimizer 종류·step 수를 동일하게 맞추고, 먼저 update 전 첫 loss가 같은지부터 확인(다르면 비교 조건이 섞였다는 신호)
- SGD: 현재 step의 gradient만 보고 모든 parameter에 같은 전역 learning rate 규칙 적용. mini-batch gradient라 noise가 포함되지만(전체 데이터 gradient보다 불안정) 이 noise가 오히려 일반화에 도움이 되기도 함. momentum을 추가하면 이전 이동 방향을 누적해 관성처럼 일관된 방향으로 밀고 좌우 진동을 줄임
- Adam: learning rate를 없애는 optimizer가 아니라 전역 learning rate에 parameter별 보정 비율을 더하는 optimizer — 1차 모멘트 `m`(최근 gradient 방향의 평균), 2차 모멘트 `v`(최근 gradient 크기 제곱의 평균)를 사용해 방향이 꾸준한 parameter는 그 흐름을 활용하고, gradient가 매우 큰 parameter는 update가 완화됨. 같은 전역 lr이어도 parameter마다 실제 update 크기가 다르게 나타남. 항상 SGD보다 좋은 건 아니고, optimizer state를 추가 저장하므로 메모리 사용량이 더 큼
- optimizer는 학습 loop 밖에서 한 번만 생성 — momentum/Adam의 `m`, `v`, step count를 step 사이에 유지하기 위함. model을 새로 만들면 optimizer도 새로 만들어야 함(변수 이름이 같아도 optimizer는 여전히 이전 parameter 객체를 참조)
- parameter group으로 층마다 다른 lr 설정 가능(fine-tuning에서 자주 사용) — 적용 전 모든 parameter가 정확히 한 group에 포함됐는지 확인 필요
- checkpoint 저장 시 모델 weight뿐 아니라 `optimizer.state_dict()`도 함께 저장해야 Adam의 누적 모멘트와 step 수까지 정확히 이어서 학습할 수 있음

**5장4강. 파라미터 업데이트 코드 흐름**
- 학습 한 step: `optimizer.zero_grad()` → `outputs = model(inputs)` → `loss = criterion(outputs, targets)` → `loss.backward()` → `optimizer.step()`
- forward와 loss는 업데이트 전 parameter로 계산되고, `optimizer.step()` 이후 다음 forward부터 새 parameter가 사용됨
- PyTorch의 gradient는 기본적으로 덮어쓰기가 아니라 누적(더하기) 방식 → `zero_grad()`를 빼먹으면 이전 gradient가 계속 쌓임
- gradient accumulation: GPU 메모리 제약으로 큰 batch를 한 번에 못 돌릴 때, 여러 micro-batch의 gradient를 누적한 뒤 한 번만 `step()` — 이때는 의도적으로 중간 `zero_grad()`를 호출하지 않으므로 기본 학습 패턴과 구분해야 함
- gradient의 의미: 어떤 weight의 gradient가 `+4.0`이면 그 weight를 늘릴수록 loss가 증가한다는 뜻 → 단순 SGD는 반대 방향(줄이는 방향)으로 이동
- momentum 없는 SGD는 `expected = weight_before - lr * weight_grad`가 실제 업데이트 결과와 일치 — `torch.allclose`로 검증 가능. 원리 이해용으로 `with torch.no_grad(): param -= lr * param.grad`처럼 직접 구현할 수도 있지만, momentum·Adam state·weight decay 등 복잡한 규칙은 optimizer에 맡기는 것이 안전
- `weight_before = model.weight.detach().clone()`처럼 업데이트 전 값을 저장할 때 `detach()`(계산 그래프에서 분리)와 `clone()`(독립적인 새 Tensor로 복사)을 모두 사용해야 함 — 하나만 빠지면 안전하게 보관되지 않거나 불필요한 그래프 연결이 남을 수 있음
- parameter가 실제로 바뀌었는지는 `torch.equal`(변경 여부), `torch.allclose`(기대값과 일치 여부), update norm(`(after-before).norm()`) 세 가지로 확인. parameter update 발생 여부와 장기적인 loss 개선은 별개 — 큰 learning rate·noisy batch·복잡한 곡면에서는 한 번의 step 뒤 같은 batch loss가 오를 수도 있음

**5장 전체 정리**
- 손실 함수는 모델 예측과 정답의 차이를 loss로 계산
- 문제 유형에 따라 MSELoss / BCEWithLogitsLoss / CrossEntropyLoss를 선택
- Optimizer는 gradient를 사용해 parameter를 업데이트하며, Learning rate가 한 번의 이동 크기를 결정
- 학습 step은 `zero_grad → forward → loss → backward → step` 순서로 작성해야 함

**한 줄 정리**: 손실 함수는 예측과 정답의 차이를 문제 유형(회귀/이진분류/다중분류/다중레이블)에 맞는 방식으로 숫자화하고, optimizer(SGD/Adam)는 그 gradient를 이용해 learning rate만큼 parameter를 이동시키며, 실제 학습 코드는 항상 `zero_grad → forward → loss → backward → step` 순서를 지켜야 한다.
