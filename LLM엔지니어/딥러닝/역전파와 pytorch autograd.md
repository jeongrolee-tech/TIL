# 2026-08-25 역전파와 PyTorch Autograd

## 오늘의 TIL 요약

**6장1강. 계산 그래프와 Chain Rule**
- gradient: 현재 위치에서 파라미터가 조금 변할 때 loss가 얼마나·어느 방향으로 변하는지를 나타내는 변화율 (실제 이동량은 optimizer + learning rate가 결정)
- `torch.autograd`: 계산 그래프를 이용해 gradient를 자동 계산하는 PyTorch의 자동미분 엔진
- forward는 입력 → 연산들 → loss로 이어지는 값 계산 흐름(왼쪽→오른쪽), backward는 loss에서 시작해 파라미터 방향으로 gradient를 전달하는 흐름(오른쪽→왼쪽)
- 계산 그래프 = "계산의 영수증" — 어떤 값이 어떤 연산을 거쳐 만들어졌는지 기록되어 있어 backward 때 거꾸로 따라갈 수 있음
- Chain Rule 예시: `c = a*b`, `d = c+1`, `loss = d**2`일 때 `d(loss)/d(a) = d(loss)/d(d) × d(d)/d(c) × d(c)/d(a) = 14 × 1 × 3 = 42` — 중간 노드를 거칠 때마다 미분을 하나씩 곱하는 것이 Chain Rule
- b는 a로 미분할 때 상수로 취급(다른 leaf tensor에 대한 편미분과 무관)

**6장2강. requires_grad와 Tensor gradient**
- `requires_grad=True`: 해당 Tensor에 대한 연산을 추적해 gradient 계산 대상으로 삼겠다는 뜻
- leaf tensor: 학습에서 실제로 업데이트되는 원재료(주로 모델의 weight/bias) — backward 후 `.grad`에 gradient가 저장됨
- non-leaf tensor: layer를 통과해 만들어진 중간 activation 등 연산 결과 — 기본적으로 `.grad`가 저장되지 않으며, 보고 싶으면 `retain_grad()` 필요
- 파라미터의 gradient shape는 파라미터 shape와 동일
- `detach()`는 값을 삭제/0으로 만드는 게 아니라 forward 값은 그대로 두고 backward가 지나갈 길만 끊음 → `detach().clone()`은 "gradient는 필요 없고 이 값만 독립적으로 저장"하겠다는 뜻
- `detach()`(특정 Tensor 하나에서 경로 차단) vs `torch.no_grad()`(블록 전체 연산에서 gradient 추적 자체를 끔, validation/inference에 사용)

**6장3강. loss.backward()와 .grad 확인**
- `loss.backward()`는 파라미터 값을 바로 바꾸지 않고 gradient를 계산해 `.grad`에 저장만 함 — 실제 업데이트는 `optimizer.step()`에서 발생
- backward 전에는 파라미터의 `.grad`가 보통 `None`
- gradient shape는 parameter shape와 동일, loss는 보통 scalar로 만든 뒤 backward 호출
- gradient가 계속 `None`이면 graph 연결 여부, detach 사용 여부, no_grad 블록, requires_grad 설정을 확인해야 함
- 다시 gradient를 구하고 싶으면 backward()를 재호출하는 게 아니라 forward를 다시 실행해 새 loss를 만든 뒤 backward 호출

**6장4강. zero_grad, backward, step 순서**
- PyTorch gradient는 기본적으로 덮어쓰기가 아니라 누적(더하기) 방식
- `optimizer.zero_grad()`: 이전 step의 gradient 초기화 → `loss.backward()`: 현재 loss 기준 gradient 계산 → `optimizer.step()`: 그 gradient로 파라미터 업데이트
- gradient 누적은 일부 고급 기법(gradient accumulation 등)에서 의도적으로 쓰이지만, 일반적인 학습 루프에서는 매 step 초기화가 기본
- 학습이 이상하면 한 step에서 파라미터가 실제로 바뀌는지부터 확인

**6장5강. Autograd 디버깅과 안전한 평가 코드**
- `model.train()` / `model.eval()`은 Dropout, BatchNorm 같은 내부 모듈의 동작 모드만 바꿈 (gradient 추적 여부와는 무관)
- `model.eval()`은 gradient 추적을 끄지 않으므로, 평가 시 gradient 추적까지 끄려면 `torch.no_grad()`를 별도로 함께 사용해야 함
- `torch.no_grad()` 블록 안에서 만든 loss는 `backward()`를 호출하는 학습용 loss가 아니라 평가 전용 loss

**6장 전체 정리**
- 계산 그래프는 forward 연산 기록, Chain Rule은 여러 연산이 연결됐을 때 영향을 단계별로 전달하는 규칙
- `requires_grad=True`인 파라미터가 gradient 계산 대상이고, `loss.backward()`가 계산 그래프를 따라 gradient를 `.grad`에 저장
- gradient는 누적되므로 매 학습 step마다 `optimizer.zero_grad()` 필요, `optimizer.step()`이 `.grad`로 파라미터를 업데이트
- 평가 코드는 `model.eval()` + `torch.no_grad()`를 함께 쓰는 것이 안전

**7장1강. Dataset과 DataLoader 역할**
- Dataset: 샘플과 label을 어떻게 보관·반환할지 정의 (`__len__`: 전체 샘플 개수, `__getitem__`: index별 샘플/label 반환)
- DataLoader: Dataset에서 샘플을 batch로 묶어주는 역할 — batching, shuffling(순서 의존 방지), iteration(`for batch in loader`), parallel loading(`num_workers`)
- `TensorDataset(X, y)`은 여러 Tensor를 첫 번째 차원 기준으로 묶어줌
- DataLoader를 새로 만들거나 전처리를 바꾼 직후에는 `next(iter(loader))`로 batch 하나를 꺼내 shape를 점검하는 습관이 학습 루프 오류를 줄여줌(매 epoch마다 할 필요는 없음)

**7장2강. TensorDataset과 CustomDataset**
- TensorDataset은 이미 Tensor로 준비된 데이터를 간단히 묶을 때 사용
- Custom Dataset이 필요한 경우: 파일에서 데이터를 읽어야 할 때, 샘플마다 전처리가 필요할 때(tokenization, image transform), label을 별도 규칙으로 만들어야 할 때(파일명에서 추출), 데이터가 Tensor로 미리 준비되어 있지 않을 때(CSV row, JSON line, audio 등)
- Custom Dataset은 `__len__`과 `__getitem__`을 구현해야 함
- 실전 프로젝트에서는 모델 코드보다 데이터 로딩 코드가 더 복잡해지는 경우가 많아 Dataset 구조를 초반에 잘 잡는 것이 중요
- Dataset을 만든 뒤에는 첫 샘플과 첫 batch의 shape, dtype, label 구조를 반드시 확인해야 함

**데일리 퀴즈 오답 노트**
- Custom Dataset이 반드시 구현해야 하는 두 메서드는 `__init__`이 아니라 `__len__`, `__getitem__` (헷갈려서 오답 — `__init__`은 필수 구현 대상이 아님)

**한 줄 정리**: PyTorch Autograd는 계산 그래프를 따라 Chain Rule로 gradient를 계산해 leaf tensor(주로 모델 파라미터)의 `.grad`에 저장하며, 학습 루프는 gradient가 누적되는 특성 때문에 `zero_grad → forward → loss → backward → step` 순서를 지켜야 하고, 평가 코드는 `model.eval()` + `torch.no_grad()`를 함께 써야 안전하며, Dataset(`__len__`/`__getitem__`)과 DataLoader(batching/shuffling)는 이 학습 루프에 데이터를 안정적으로 공급하는 역할을 한다.
