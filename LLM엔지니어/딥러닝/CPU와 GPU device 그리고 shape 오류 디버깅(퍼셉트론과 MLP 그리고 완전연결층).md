# 2026-08-20 CPU/GPU device와 shape 오류 디버깅

## 오늘의 TIL 요약

**2장3강. CPU/GPU device와 .to(device)**
- device는 Tensor가 저장된 위치를 의미 (cpu / cuda, cuda:0)
- GPU를 켜도 Tensor가 자동으로 GPU로 이동하지 않음, `.to(device)`를 직접 호출해야 함
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- `Tensor.to(device)`는 이동된 새 Tensor를 반환하므로 `x = x.to(device)`처럼 재할당 필요. 반면 `nn.Module.to(device)`는 파라미터/버퍼를 이동시키고 모듈 자신을 반환
- 학습 루프에서는 model, input(x), label(y)이 모두 같은 device에 있어야 함 — device mismatch는 실무에서도 흔한 오류(images는 GPU, labels는 CPU에 남는 경우 등)

**2장4강. shape/device 오류 디버깅**
- Tensor는 shape, dtype, device를 함께 확인해야 함
- `nn.Linear`는 입력 마지막 차원이 `in_features`와 같아야 함
- `nn.Linear`는 배치 차원 없는 입력 `(4,)`도 처리 가능하지만, `unsqueeze(0)`으로 배치 차원 `(1, 4)`를 명시하는 것이 안전
- 모델 학습 시 입력/출력의 첫 번째 차원은 가능한 한 batch로 유지

**3장1강~3장5강은 이해만 하고 실습으로 바로 넘어간다**

**데일리퀴즈 정리**
1. `x.to(device)` 결과를 재할당하지 않으면 x는 그대로 CPU에 남음
   - `Tensor.to()`는 in-place가 아니라 이동된 새 텐서를 반환하는 함수이므로, `x = x.to(device)`처럼 반드시 재할당해야 실제로 device가 바뀜
   - 그냥 `x.to(device)`만 호출하고 끝내면 반환값이 버려지고 원래 x는 그대로 이전 device에 남아있음 — 파이썬 문법상 에러가 나지 않기 때문에 눈치채기 어려운 실수
   - 이 실수가 실무에서 흔한 이유는, model을 `.to(device)`한 뒤 학습 루프 안에서 `images.to(device)`처럼 재할당 없이 쓰는 경우 model과 input의 device가 어긋나면서 `RuntimeError: Expected all tensors to be on the same device` 오류로 이어지기 때문
   - 반대로 `model.to(device)`는 `nn.Module`의 메서드라 파라미터/버퍼를 in-place로 이동시키고 self(모듈 자신)를 반환하므로 재할당 없이 `model.to(device)`만 호출해도 동작함 — Tensor와 Module의 `.to()` 동작 차이를 구분해서 기억해야 함

2. `CrossEntropyLoss`의 class index target dtype은 `long`(int64)
   - target을 float으로 넘기면 `RuntimeError: expected scalar type Long but found Float` 같은 dtype 오류가 발생
   - label 텐서를 만들 때 `.long()`으로 명시적으로 캐스팅해줘야 하며, `torch.tensor(labels, dtype=torch.long)`처럼 생성 시점에 dtype을 지정해도 됨
   - `CrossEntropyLoss`는 target으로 one-hot 벡터가 아니라 클래스 인덱스(정수)를 받는 방식이라, 내부적으로 `LogSoftmax + NLLLoss`를 함께 수행함 — 그래서 모델의 마지막 층 출력(logits)에 별도로 softmax를 적용하면 안 되고 raw logits을 그대로 넘겨야 함
   - target dtype이 long이어야 하는 이유는 이 값이 실수 연산에 쓰이는 게 아니라 정답 클래스를 가리키는 인덱스로 사용되기 때문(정수형 인덱싱)

3. `flatten(images, start_dim=1)`은 batch 차원 유지 → `(16,1,28,28)` → `(16,784)`
   - `start_dim` 인자는 몇 번째 차원부터 하나로 합칠지를 정하는 값이라, `start_dim=1`이면 0번째(batch) 차원은 그대로 두고 1번째 차원부터 끝까지만 평탄화됨
   - 만약 `start_dim=0`으로 주면 batch 차원까지 합쳐져 `(16,1,28,28)` 전체가 `(12544,)` 하나의 벡터가 되어버려 배치 정보 자체가 사라지는 실수로 이어짐
   - MNIST 같은 이미지 데이터를 완전연결층(`nn.Linear`)에 넣기 전에는 `(batch, channel, height, width)` 형태를 `(batch, channel*height*width)`로 펴줘야 하므로 `flatten(x, start_dim=1)` 또는 `x.view(x.size(0), -1)`을 관용적으로 사용

4. `nn.Linear(10,5)`의 weight shape는 `(5,10)` — (out_features, in_features) 순서
   - 내부 계산은 `input @ weight.T + bias` 형태이며, bias shape는 `(5,)`로 out_features와 같음
   - weight 순서를 (in, out)으로 착각하면 직접 행렬곱을 구현할 때 shape mismatch 오류의 원인이 되므로, PyTorch는 (out_features, in_features) 순서로 저장한다는 점을 기억해야 함
   - `model.weight.shape`, `model.bias.shape`로 직접 찍어보면서 확인하는 습관이 shape 디버깅에 도움이 됨

5. MLP를 통과하며 batch size는 유지되고 feature 차원만 변함
   - 각 층을 지날 때마다 배치 차원(첫 번째 차원)은 고정된 채로 feature 차원만 `in_features → out_features`로 바뀌어야 정상 (예: `(16,784) → (16,256) → (16,128) → (16,10)`)
   - `nn.ReLU` 같은 활성화 함수는 shape에 영향을 주지 않고 값만 변형하므로, shape 오류가 났다면 활성화 함수가 아니라 `nn.Linear`의 in/out features 설정이 이전 층의 출력과 맞는지를 먼저 의심해야 함
   - 배치 차원이 중간에 바뀌거나 사라진다면 이는 대부분 `flatten`/`view`/`reshape`를 잘못된 `dim` 기준으로 호출했기 때문

**한 줄 정리**: device는 자동 이동이 안 되므로 `.to(device)` 결과를 재할당해 model/입력/라벨의 device를 통일해야 하고, shape 오류 디버깅에서는 batch 차원 유지가 핵심.
