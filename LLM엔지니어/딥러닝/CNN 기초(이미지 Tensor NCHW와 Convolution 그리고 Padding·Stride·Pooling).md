# 2026-08-28 CNN 기초(이미지 Tensor NCHW와 Convolution 그리고 Padding·Stride·Pooling)

## 오늘의 TIL 요약

**11장1강. 이미지 데이터 구조와 channel**
- 지금까지 MLP에서 다룬 입력은 `(batch_size, features)`의 **2차원** 텐서였지만, 이미지는 공간 구조(위/아래, 좌/우)를 가지므로 **4차원** 텐서로 다룸
  - PyTorch의 표준 이미지 batch 레이아웃은 **`(N, C, H, W)`** — 줄여서 **NCHW**
  | 축 | 이름 | 의미 |
  | --- | --- | --- |
  | N | batch_size | 한 번에 처리할 이미지 장수 |
  | C | channels | 색 성분 수 (RGB=3, 흑백=1) |
  | H | height | 이미지 세로 픽셀 수 |
  | W | width | 이미지 가로 픽셀 수 |
  ```python
  import torch

  # RGB 이미지 8장, 각 32x32 크기라고 가정
  images = torch.randn(8, 3, 32, 32)

  print("images shape:", images.shape)   # torch.Size([8, 3, 32, 32])

  # 각 차원을 변수로 풀어 읽으면 shape의 의미가 훨씬 명확해짐
  batch_size, channels, height, width = images.shape
  print(batch_size, channels, height, width)   # 8 3 32 32
  ```
- **왜 channel 수가 중요한가**: CNN은 이미지를 한 줄로 펴지 않고 **작은 영역을 훑으며** 패턴을 찾는데, RGB 이미지는 한 픽셀 안에 R/G/B 세 값이 겹쳐 있음 → 첫 Conv layer는 **입력 channel 수를 정확히 알아야** 그 세 값을 함께 계산할 수 있음
- ⛔ **NCHW vs HWC 혼동이 CNN 첫 에러의 대부분**
  - 일반 이미지 처리 라이브러리(PIL, OpenCV, matplotlib 등)는 보통 **`(H, W, C)`** 순서로 배열을 다룸
  - 반대로 PyTorch `Conv2d`는 **`(N, C, H, W)`**를 요구 → 그대로 넣으면 channel 개수가 안 맞아 런타임 에러
  ```python
  # HWC 형태의 이미지 한 장 (일반 라이브러리에서 흔한 순서)
  hwc_image = torch.randn(32, 32, 3)
  print("HWC:", hwc_image.shape)          # torch.Size([32, 32, 3])

  # permute(2, 0, 1): (H=0, W=1, C=2) -> (C, H, W) 로 축 순서를 재배치
  chw_image = hwc_image.permute(2, 0, 1)
  print("CHW:", chw_image.shape)          # torch.Size([3, 32, 32])

  # Conv2d에 넣으려면 batch 차원도 필요 -> unsqueeze(0)으로 맨 앞에 N=1 추가
  nchw_image = chw_image.unsqueeze(0)
  print("NCHW:", nchw_image.shape)        # torch.Size([1, 3, 32, 32])
  ```
- ⛔ `permute`는 **축의 순서만 바꾸는 view 연산** — 값을 정규화하거나 크기를 바꾸지 않고, 데이터 복사도 하지 않음
  - 대신 결과가 메모리에서 **연속(contiguous)하지 않을 수 있음** → 이후 `view`를 쓰면 에러가 날 수 있으므로 `.contiguous()`를 붙이거나, 애초에 `view` 대신 `reshape`을 사용
  - `permute`(축 재배치)와 `reshape`/`view`(원소 개수를 유지한 채 shape 변경)는 **역할이 완전히 다름** — `reshape(3, 32, 32)`로 HWC를 CHW로 바꾸면 픽셀이 뒤섞여 이미지가 망가짐
- 참고: `torchvision.transforms.ToTensor()`는 PIL 이미지를 읽으면서 **HWC → CHW 변환과 0~255 → 0.0~1.0 스케일링을 함께** 수행 → DataLoader를 쓰면 대부분 이 변환이 자동으로 처리됨
- 💡 이미지 shape가 헷갈릴 때의 첫 번째 행동은 언제나 **`print(x.shape)`**

**11장2강. Convolution, Kernel, Filter**
- **CNN(Convolutional Neural Network)** = 이미지·영상처럼 **공간적 구조**를 가진 데이터를 처리하는 신경망
  - 이미지를 처음부터 한 줄로 펴지 않고, 작은 창(filter)을 이미지 전체에 반복 적용해 선·경계·질감 같은 **지역 패턴**을 찾음
- CNN을 떠받치는 두 가지 핵심 아이디어
  | 아이디어 | 의미 | 얻는 것 |
  | --- | --- | --- |
  | 국소 연결(local connectivity) | 한 번에 이미지 전체가 아니라 작은 영역만 봄 | 공간 구조 보존, 연결 수 감소 |
  | 가중치 공유(weight sharing) | 같은 filter를 모든 위치에 반복 적용 | parameter 급감, 위치가 달라도 같은 패턴을 같은 방식으로 탐지 |
- **왜 MLP 대신 CNN인가** — parameter 수를 직접 비교해보면 차이가 분명함
  | 방식 | 구성 | parameter 수 |
  | --- | --- | --- |
  | MLP | `(3,32,32)` flatten(3072) → `nn.Linear(3072, 32)` | 3072×32 + 32 = **98,336** |
  | CNN | `nn.Conv2d(3, 32, kernel_size=3, padding=1)` | 3×3×3×32 + 32 = **896** |
  - 게다가 MLP는 flatten하는 순간 "이 픽셀 옆에 저 픽셀이 있었다"는 **공간 정보가 사라지고**, 입력 H,W가 바뀌면 `in_features`도 전부 바뀜
  - Conv의 parameter 수는 **입력 H,W와 무관** — 같은 filter를 위치만 옮겨 쓰기 때문
- 용어 정리
  - **kernel / filter** = 이미지에서 찾고 싶은 작은 패턴을 검사하는 도구 (예: 3×3 창)
  - **feature map** = 그 패턴이 이미지의 **어느 위치에서 얼마나 강하게** 나타났는지를 기록한 결과
  - filter 값은 사람이 정하지 않고 **역전파로 학습됨** → 여러 filter를 쓰면 서로 다른 패턴에 반응하는 여러 feature map을 얻음
- Convolution의 실제 계산: 창 안의 값들과 kernel 값을 **원소별로 곱해서 모두 더한 뒤**(가중합) bias를 더함 → 그 결과가 출력의 한 칸
  ```python
  import torch
  import torch.nn.functional as F

  # 세로 경계선을 찾는 3x3 kernel 예시 (실제로는 이 값도 학습됨)
  x = torch.tensor([[[[0., 0., 1., 1.],
                      [0., 0., 1., 1.],
                      [0., 0., 1., 1.],
                      [0., 0., 1., 1.]]]])            # (N=1, C=1, H=4, W=4)

  kernel = torch.tensor([[[[-1., 0., 1.],
                           [-1., 0., 1.],
                           [-1., 0., 1.]]]])          # (out=1, in=1, kh=3, kw=3)

  y = F.conv2d(x, kernel)
  print(y)          # 값이 0에서 1로 바뀌는 경계 위치에서만 큰 값이 나옴
  print(y.shape)    # torch.Size([1, 1, 2, 2])
  ```
- **RGB 입력에서 filter 하나가 하는 일**: R, G, B 세 channel의 같은 위치 영역을 **각각 계산한 뒤 전부 더해** 출력 channel **한 칸**을 만듦
  - 즉 filter는 `(in_channels, kh, kw)` 크기의 **3차원 블록**이고, 이런 블록이 `out_channels`개 있음
- **`nn.Conv2d`의 핵심 인자**
  | 인자 | 의미 | 예시 |
  | --- | --- | --- |
  | `in_channels` | 입력 이미지/feature map의 channel 수 | RGB면 3 |
  | `out_channels` | 만들어낼 feature map 수 (= filter 개수) | 8, 16, 32 … |
  | `kernel_size` | filter의 공간 크기 | 3 → 3×3 |
  | `padding` | 입력 가장자리에 덧붙일 두께 | 1 |
  | `stride` | kernel이 한 번에 이동하는 간격 | 1 또는 2 |
  | `bias` | channel별 bias 사용 여부 | BatchNorm을 붙이면 `False`로 두기도 함 |
- **weight shape는 `(out_channels, in_channels, kernel_height, kernel_width)`**
  ```python
  import torch.nn as nn

  conv = nn.Conv2d(3, 8, kernel_size=3)
  print(conv.weight.shape)   # torch.Size([8, 3, 3, 3])
  print(conv.bias.shape)     # torch.Size([8])
  ```
  - `nn.Linear`의 weight가 `(out_features, in_features)`였던 것처럼 **출력 쪽이 먼저** 온다고 기억하면 편함
  - parameter 수 = `out_channels × in_channels × kh × kw + out_channels`
- 출력 텐서 shape는 여전히 **`(N, out_channels, H_out, W_out)`** — Conv는 channel 수를 바꾸고, H·W는 padding/stride가 결정
- ⛔ 자주 하는 실수
  | 실수 | 증상 | 해결 |
  | --- | --- | --- |
  | 입력 channel과 `in_channels` 불일치 | `expected input[...] to have 3 channels, but got 1 channels instead` | 흑백이면 `in_channels=1`, RGB면 3 |
  | `out_channels`를 class 수로 오해 | 구조가 이상해지거나 마지막 Linear가 안 맞음 | 중간 `out_channels`는 **feature map 개수**, class 수는 **마지막 `nn.Linear`의 `out_features`** |
  | HWC를 그대로 입력 | channel 개수 에러 | `permute(2,0,1)` 후 `unsqueeze(0)` |
- (심화) **PyTorch의 `Conv2d`는 엄밀히 말하면 cross-correlation** — 수학적 convolution과 달리 kernel을 뒤집지 않고 곱함. kernel 값 자체를 학습하므로 뒤집든 말든 표현력이 같아, 딥러닝에서는 관례적으로 convolution이라고 부름
- (심화) **1×1 convolution**: `nn.Conv2d(64, 32, kernel_size=1)`은 H,W를 그대로 두고 **channel 수만 섞어서 조절**하는 용도 → 채널 압축·확장에 자주 쓰임

**CNN의 전체 흐름**
```
입력 이미지 (N, C, H, W)
  → Convolution : 지역 패턴을 찾아 feature map 생성 (channel 변경)
  → ReLU        : 비선형성 추가
  → Pooling     : feature map의 공간 크기(H, W) 축소
  → (Conv/ReLU/Pooling 반복) : 점점 더 복잡하고 추상적인 특징 추출
  → Flatten     : feature map을 벡터로 변환
  → Linear      : class별 logits 출력 (N, num_classes)
```
- 앞부분(Conv/ReLU/Pool)은 **feature extractor**, 뒷부분(Flatten/Linear)은 **classifier**
- 그림에 Softmax가 그려져 있더라도, PyTorch에서 `CrossEntropyLoss`를 쓸 때는 모델이 **softmax 이전의 logits를 그대로 출력**해야 함 (손실 함수 내부에서 log-softmax를 처리)
- CNN의 국소 수용영역(local receptive field)과 가중치 공유는 시각 정보 처리 방식에서 **아이디어를 얻은 것**일 뿐, 사람의 시신경을 그대로 복제한 모델은 아님. 이미지 분류·객체 탐지·분할·자율주행 등에서 널리 쓰임

**11장3강. Padding과 Stride**
- **Padding** = 입력 가장자리 바깥에 값을 덧붙이는 것. 보통 0을 채우므로 **zero padding**
  | padding이 필요한 이유 | 설명 |
  | --- | --- |
  | 가장자리 정보 보존 | padding이 없으면 모서리 픽셀은 kernel에 몇 번 포함되지 않아 정보가 빨리 사라짐 |
  | 공간 크기 유지 | Conv를 여러 층 쌓아도 H,W가 급격히 줄지 않게 함 |
  - `kernel_size=3, stride=1, padding=1` 조합은 **H,W를 그대로 유지**하는 대표 레시피 (`same` 크기)
- **Stride** = kernel이 한 번에 건너뛰는 칸 수. stride가 커지면 kernel이 방문하는 위치 수가 줄어 **출력 H,W가 작아짐**
- ⭐ **출력 크기 공식** (가장 자주 쓰는 형태, `dilation=1` 기준)
  ```
  H_out = floor( (H_in + 2*P - K) / S ) + 1
  W_out = floor( (W_in + 2*P - K) / S ) + 1
  ```
  - H와 W는 **각각 따로** 계산해야 함 (정사각형이 아니면 결과가 다름)
  - `dilation`까지 포함한 일반형: `H_out = floor((H_in + 2P − D×(K−1) − 1) / S) + 1`
  - `floor`(내림)가 붙는 이유: kernel이 입력 밖으로 나가는 위치는 아예 계산하지 않고 **버리기** 때문
- 대표 조합 정리 (`H_in = 32` 기준)
  | K | P | S | H_out | 효과 |
  | --- | --- | --- | --- | --- |
  | 3 | 1 | 1 | 32 | 크기 유지 (가장 흔한 기본형) |
  | 3 | 0 | 1 | 30 | 매 층 2씩 감소 |
  | 5 | 2 | 1 | 32 | 크기 유지 (더 넓은 시야) |
  | 3 | 1 | 2 | 16 | 절반으로 축소 |
  | 7 | 3 | 2 | 16 | 절반 축소 + 넓은 시야 |
  - 규칙: **stride=1이고 kernel이 홀수일 때 `padding = (K−1)/2`이면 H,W가 유지됨**
  ```python
  import torch
  import torch.nn as nn

  x = torch.randn(1, 3, 32, 32)

  conv_s1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1)
  conv_s2 = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2)

  print("stride=1:", conv_s1(x).shape)   # torch.Size([1, 8, 32, 32])
  print("stride=2:", conv_s2(x).shape)   # torch.Size([1, 8, 16, 16])
  ```
- ⛔ 자주 하는 실수
  1. **"padding=1이면 항상 크기가 유지된다"고 외우기** — 유지되는 것은 `K=3, S=1, P=1`일 때. `K=5, P=1, S=1`이면 32 → 30으로 줄어듦
  2. **H와 W를 따로 확인하지 않기** — 정사각형이 아닌 이미지는 두 축이 다르게 변함
     ```python
     x = torch.randn(1, 3, 64, 32)      # H=64, W=32
     conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2)
     print(conv(x).shape)               # torch.Size([1, 8, 32, 16])  <- H와 W가 각각 절반
     ```
  3. **stride를 크게 줘서 정보를 과하게 버리기** — 초반부터 `stride=4` 같은 값을 쓰면 세밀한 패턴이 사라짐
- (참고) `padding="same"` 문자열도 지원되지만 **`stride=1`일 때만** 사용 가능. 숫자로 직접 지정하는 편이 shape 계산을 눈으로 따라가기 좋음
- **연습 문제**: 입력 `(1, 3, 32, 32)`에 `nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=1)`을 적용하면?
  ```
  N = 1, out_channels = 16, H_in = W_in = 32, K = 5, P = 2, S = 1

  H_out = floor( (32 + 2*2 - 5) / 1 ) + 1 = floor(31) + 1 = 32
  W_out = 32  (동일)

  → 출력 shape = (1, 16, 32, 32)
  ```
- (심화) **receptive field(수용영역)** — 3×3 conv를 **두 번** 쌓으면 출력 한 칸이 원본의 5×5 영역을 보게 됨. parameter는 `3×3×2 = 18`로 `5×5 = 25`보다 적으면서 비선형(ReLU)을 한 번 더 넣을 수 있어, 요즘 CNN이 **작은 kernel을 깊게 쌓는** 이유가 됨

**11장4강. Pooling 연산**
- **Pooling** = feature map의 **공간 크기(H, W)를 줄이는** 연산. 가장 자주 쓰는 것은 **Max Pooling**(작은 영역에서 가장 큰 값만 남김)
- ⭐ **Pooling에는 학습 parameter가 없음** — weight나 bias가 없는 고정된 규칙 연산이라 `model.parameters()`에 잡히지 않음
  ```python
  import torch
  import torch.nn as nn

  x = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                      [ 5.,  6.,  7.,  8.],
                      [ 9., 10., 11., 12.],
                      [13., 14., 15., 16.]]]])   # (N=1, C=1, H=4, W=4)

  # stride를 생략하면 kernel_size와 같은 값이 기본으로 사용됨 -> 2x2 영역이 겹치지 않게 이동
  pool = nn.MaxPool2d(kernel_size=2)
  y = pool(x)

  print(y)          # tensor([[[[ 6.,  8.], [14., 16.]]]])
  print(y.shape)    # torch.Size([1, 1, 2, 2])
  ```
  - 각 2×2 블록의 최댓값: 좌상 `{1,2,5,6}` → 6, 우상 `{3,4,7,8}` → 8, 좌하 `{9,10,13,14}` → 14, 우하 `{11,12,15,16}` → 16
- **왜 pooling을 쓰는가**
  | 목적 | 설명 |
  | --- | --- |
  | 공간 크기 감소 | H,W를 줄여 이후 층의 계산량과 메모리를 줄임 |
  | 중요한 반응 유지 | MaxPool은 "가장 강하게 반응한" 값만 남김 |
  | 작은 위치 변화에 둔감 | 특징이 몇 픽셀 이동해도 비슷한 출력 (local translation invariance) |
  | receptive field 확대 | 같은 kernel이 더 넓은 원본 영역을 보게 됨 |
- ⭐ **Pooling은 channel 수를 바꾸지 않음** — 각 channel 안에서 **독립적으로** 공간 축만 줄임
  - `(8, 16, 32, 32)` → `MaxPool2d(2)` → `(8, 16, 16, 16)`  (N=8, C=16 그대로)
- Pooling 출력 크기도 Conv와 **같은 공식**을 사용: `H_out = floor((H_in + 2P − K) / S) + 1`
  - `MaxPool2d(2)`는 `K=2, S=2, P=0` → 짝수 H,W면 정확히 절반
  - **홀수일 때 주의**: `H_in=7` → `floor((7−2)/2)+1 = 3`으로 한 줄이 버려짐 → 버리기 싫으면 `ceil_mode=True`(→4)
- Pooling 종류 비교
  | 종류 | 동작 | 특징 |
  | --- | --- | --- |
  | `nn.MaxPool2d` | 영역의 최댓값 | 가장 강한 반응 보존, 경계·질감에 강함 (기본 선택) |
  | `nn.AvgPool2d` | 영역의 평균 | 전체적으로 부드럽게, 강한 반응이 희석됨 |
  | `nn.AdaptiveAvgPool2d((1,1))` | 각 channel 전체를 평균내 1×1로 | **Global Average Pooling** — 입력 H,W와 무관하게 `(N, C, 1, 1)` 출력 |
  - 최근 CNN은 pooling 대신 **stride=2 convolution**으로 크기를 줄이기도 함(줄이는 방법 자체를 학습시키는 셈)
- **CNN block에서의 사용** — `Conv → ReLU → Pool`이 기본 단위
  ```python
  x = torch.randn(8, 3, 32, 32)

  block = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, padding=1),   # (8, 3, 32, 32) -> (8, 16, 32, 32)
      nn.ReLU(),                                    # shape 변화 없음
      nn.MaxPool2d(kernel_size=2),                  # (8, 16, 32, 32) -> (8, 16, 16, 16)
  )

  print("input :", x.shape)          # torch.Size([8, 3, 32, 32])
  print("output:", block(x).shape)   # torch.Size([8, 16, 16, 16])
  ```
  - 순서 관례: **`Conv → BatchNorm2d → ReLU → Pool`** (BatchNorm을 넣을 경우). `BatchNorm2d`의 인자는 **channel 수**(`BatchNorm1d`가 feature 수를 받았던 것과 대응)
- ⛔ pooling을 남발하면 H,W가 순식간에 1까지 줄어듦 — `32 → 16 → 8 → 4 → 2 → 1`. 층을 쌓을수록 **H,W는 줄이고 channel은 늘리는**(예: 3 → 32 → 64 → 128) 패턴이 일반적
- 💬 Pooling은 단순한 축소가 아니라, **CNN이 중요하다고 판단한 반응을 남기면서 압축하는** 과정으로 이해하면 좋음

**11장5강. Feature map shape 확인**
- CNN에서 shape 에러가 가장 많이 터지는 지점은 **feature extractor의 마지막 → Linear로 넘어가는 경계** → 중간 feature map shape를 **계속** 확인하는 습관이 핵심
- **Flatten**: `(N, C, H, W)` → `(N, C×H×W)`
  ```python
  # CNN 출력이 (batch, 64, 4, 4)라면
  # flatten_dim = 64 * 4 * 4 = 1024
  x = torch.flatten(x, start_dim=1)   # (batch, 1024)  <- batch 차원(0번)은 유지
  fc = nn.Linear(1024, 10)            # in_features == flatten_dim == 1024
  ```
  | 방법 | 특징 |
  | --- | --- |
  | `torch.flatten(x, start_dim=1)` | 함수형. batch 차원을 남기고 나머지를 모두 펼침 |
  | `nn.Flatten()` | `nn.Sequential` 안에 넣을 수 있는 layer 버전. 기본이 `start_dim=1` |
  | `x.view(x.size(0), -1)` | 빠르지만 **contiguous하지 않으면 에러** |
  | `x.reshape(x.size(0), -1)` | 필요하면 내부에서 복사해줘서 더 안전 |
  - ⛔ `torch.flatten(x)`처럼 `start_dim`을 빼면 **batch까지 섞여** `(N×C×H×W,)` 한 줄이 되어버림
- **shape 추적표**를 손으로 한 번 그려두면 flatten_dim 계산 실수가 거의 사라짐 (입력 `(N, 3, 32, 32)` 기준)
  | 단계 | 연산 | 출력 shape |
  | --- | --- | --- |
  | 입력 | — | `(N, 3, 32, 32)` |
  | block1 | `Conv2d(3,32,3,p=1)` + ReLU | `(N, 32, 32, 32)` |
  |  | `MaxPool2d(2)` | `(N, 32, 16, 16)` |
  | block2 | `Conv2d(32,64,3,p=1)` + ReLU | `(N, 64, 16, 16)` |
  |  | `MaxPool2d(2)` | `(N, 64, 8, 8)` |
  | block3 | `Conv2d(64,128,3,p=1)` + ReLU | `(N, 128, 8, 8)` |
  |  | `MaxPool2d(2)` | `(N, 128, 4, 4)` |
  | flatten | `flatten(start_dim=1)` | `(N, 2048)` ← 128×4×4 |
  | classifier | `Linear(2048, 10)` | `(N, 10)` |
- ⭐ **더미 입력으로 flatten_dim을 자동 계산하기** — 구조를 바꿀 때마다 손계산하지 않아도 됨
  ```python
  feature_extractor = nn.Sequential(
      nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
      nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
  )

  def get_flatten_dim(module, input_shape=(3, 32, 32), device="cpu"):
      was_training = module.training
      module.eval()                              # BatchNorm running stats 오염 방지
      with torch.no_grad():                      # gradient 그래프도 만들지 않음
          dummy = torch.zeros(1, *input_shape, device=device)   # 실제 입력과 device/dtype 일치
          out = module(dummy)
      module.train(was_training)                 # 원래 모드로 되돌리기
      return out.flatten(1).shape[1]

  flatten_dim = get_flatten_dim(feature_extractor)
  print(flatten_dim)   # 2048
  ```
  - 더미 입력을 쓸 때 주의할 점 세 가지: ① **`model.eval()`** — train 모드면 BatchNorm의 running mean/var가 더미 값으로 갱신됨, ② **`torch.no_grad()`**, ③ **device·dtype을 실제 입력과 맞추기**
- ⭐ **애초에 flatten_dim을 고정하는 방법 — Global Average Pooling**
  ```python
  # 입력 H,W가 무엇이든 (N, 128, 1, 1)로 만들어줌 -> in_features는 항상 channel 수(128)
  head = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(128, 10),
  )
  ```
  - 입력 크기가 바뀌어도 Linear를 다시 계산할 필요가 없고, parameter 수도 크게 줄어듦
- ⛔ 대표적인 shape 에러와 원인
  | 에러 메시지(요지) | 원인 | 해결 |
  | --- | --- | --- |
  | `mat1 and mat2 shapes cannot be multiplied (8x1024 and 2048x10)` | flatten_dim ≠ Linear의 `in_features` | Conv/Pool 구조를 바꿨으면 flatten_dim도 다시 계산 |
  | `expected input[...] to have 3 channels, but got 1 channels` | 입력 channel ≠ `in_channels` | 흑백/RGB 확인, HWC→CHW 변환 확인 |
  | `view size is not compatible ... use .reshape()` | `permute` 이후 non-contiguous 상태에서 `view` | `.contiguous()` 또는 `reshape` 사용 |
  | 학습은 되는데 accuracy가 이상함 | batch 차원까지 flatten | `start_dim=1` 확인 |
  - `Conv2d`는 batch 없는 3차원 입력 `(C,H,W)`도 받아주지만, 그 뒤 `flatten(start_dim=1)`부터 의미가 어긋남 → **입력에 batch 차원이 있는지 먼저 확인**
- 💡 디버깅 팁: `nn.Sequential` 사이사이에 shape를 찍어보는 것이 가장 빠름
  ```python
  x = torch.randn(2, 3, 32, 32)
  for i, layer in enumerate(feature_extractor):
      x = layer(x)
      print(i, layer.__class__.__name__, tuple(x.shape))
  ```

**11장6강. CNN classifier 구조 연결**
- 지금까지의 Conv / ReLU / Pooling / Flatten / Linear를 이어 붙여 **이미지 분류기**를 완성
  | 부분 | 역할 | 구성 |
  | --- | --- | --- |
  | **feature extractor** | 이미지에서 공간적 특징 추출 | `Conv2d` + `ReLU` + `MaxPool2d` 반복 |
  | **classifier** | 추출된 특징을 class별 logits로 변환 | `Flatten` + `Linear` (+ `Dropout`) |
  ```python
  import torch
  import torch.nn as nn

  class SimpleCNN(nn.Module):
      def __init__(self, in_channels=3, num_classes=10, flatten_dim=128 * 4 * 4):
          super().__init__()

          # ---- feature extractor: H,W는 줄이고 channel은 늘림 ----
          self.features = nn.Sequential(
              # (N, 3, 32, 32) -> (N, 32, 16, 16)
              nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
              nn.BatchNorm2d(32),          # 인자는 batch size가 아니라 channel 수
              nn.ReLU(),
              nn.MaxPool2d(2),

              # (N, 32, 16, 16) -> (N, 64, 8, 8)
              nn.Conv2d(32, 64, kernel_size=3, padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(),
              nn.MaxPool2d(2),

              # (N, 64, 8, 8) -> (N, 128, 4, 4)
              nn.Conv2d(64, 128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.MaxPool2d(2),
          )

          # ---- classifier: 특징 벡터 -> class별 점수 ----
          self.classifier = nn.Sequential(
              nn.Flatten(),                        # (N, 128, 4, 4) -> (N, 2048)
              nn.Dropout(0.3),
              nn.Linear(flatten_dim, num_classes)  # (N, 2048) -> (N, 10), softmax 없음
          )

      def forward(self, x):
          x = self.features(x)
          x = self.classifier(x)
          return x                                 # logits

  model = SimpleCNN(in_channels=3, num_classes=10)
  x = torch.randn(8, 3, 32, 32)
  logits = model(x)

  print(logits.shape)   # torch.Size([8, 10])  = (batch_size, num_classes)
  ```
- ⭐ **분류 모델의 최종 출력은 `(batch_size, num_classes)` 형태의 logits** — `CrossEntropyLoss`를 쓸 때는 **softmax를 미리 적용하지 않음**
  - 손실 함수 내부에서 log-softmax를 처리하므로, softmax를 두 번 걸면 gradient가 뭉개져 학습이 잘 안 됨
  - 확률이 필요할 때만 **추론 시점에** `torch.softmax(logits, dim=1)`, 예측 class는 `logits.argmax(dim=1)`
- 학습 루프는 MLP 때와 **완전히 동일** — 바뀐 것은 모델 내부뿐이고, `criterion`/`optimizer`/`train()`·`eval()`/checkpoint 관리는 그대로 재사용
  ```python
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  model.train()
  for images, labels in train_loader:            # images: (N, C, H, W), labels: (N,)
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      loss = criterion(model(images), labels)    # logits를 그대로 전달
      loss.backward()
      optimizer.step()
  ```
- 구조를 바꿀 때의 체크리스트
  1. 첫 `Conv2d`의 `in_channels` = 이미지 channel 수인가? (흑백 1, RGB 3)
  2. 각 block의 `in_channels` = 직전 block의 `out_channels`인가?
  3. Pooling 횟수를 바꿨다면 **flatten_dim을 다시 계산**했는가?
  4. 마지막 `Linear`의 `out_features` = **class 수**인가?
  5. 출력이 logits인가? (모델 안에 softmax를 넣지 않았는가)

**11장 전체 정리**
- 이미지는 **`(N, C, H, W)`** 형태로 CNN에 들어가고, HWC 배열은 `permute(2,0,1)` + `unsqueeze(0)`으로 변환
- `Conv2d`는 작은 filter로 **지역 패턴**을 찾아 feature map을 만들며, weight shape는 **`(out_channels, in_channels, kh, kw)`**
- **padding**은 가장자리를 보존하고 **stride**는 이동 간격 — 출력 크기는 `floor((H_in + 2P − K)/S) + 1`
- **MaxPool2d**는 H,W를 줄이고 강한 반응만 남기며 **channel 수와 batch 크기는 바꾸지 않고, 학습 parameter도 없음**
- CNN classifier는 **feature extractor + classifier**로 나눠 생각하고, 그 경계인 **flatten_dim**이 shape 에러의 주범
- 최종 출력은 **`(batch_size, num_classes)` logits** — `CrossEntropyLoss` 앞에서는 softmax를 걸지 않음

**데일리 퀴즈 정리**
1. MLP에서 `nn.BatchNorm1d(hidden_dim)`의 인자가 의미하는 것은? → **직전 layer의 출력 feature 수**. `fc1`의 출력이 `(batch_size, hidden_dim)`이므로 `BatchNorm1d`도 `hidden_dim`을 받아야 shape가 맞음. batch size·class 수·데이터셋 크기와는 무관
   - 같은 원리로 **`nn.BatchNorm2d`의 인자는 channel 수** — 입력이 `(N, C, H, W)`이므로 `C`를 넣음
2. `nn.Dropout`이 `model.eval()` 모드일 때의 동작은? → **비활성화되어 입력이 그대로 통과**(일부 값을 0으로 만들지 않음). `model.train()`에서만 무작위로 값을 0으로 만들기 때문에, eval 모드에서는 같은 입력에 항상 같은 출력이 나옴
3. `EarlyStopping(patience=3, min_delta=0.001)`에서 학습이 멈추는 조건은? → **validation loss가 0.001 이상 개선되지 않는 epoch가 3번 연속 발생하면 중단**. `patience`는 개선이 없어도 기다릴 epoch 수, `min_delta`는 개선으로 인정할 최소 변화폭 (train loss 기준이나 학습률 감소는 다른 개념)
4. `nn.MaxPool2d(kernel_size=2)`를 `(N, C, H, W)`에 적용했을 때 값이 그대로 유지되는 차원은? → **C (channel)**
   - MaxPool2d는 **각 channel 안에서 공간 영역(H, W)만** 줄이는 연산 → `(8, 16, 32, 32)` → `(8, 16, 16, 16)`으로 channel 16이 그대로 유지됨
   - batch 차원 N도 값이 바뀌지 않지만, 이 문항이 묻는 것은 **"pooling이 무엇을 건드리고 무엇을 건드리지 않는가"** → pooling이 실제로 작용하는 대상은 H·W이고 그 연산이 **channel별로 독립 수행**되므로 답은 C
   - 정리 문장: **"Pooling은 H,W만 줄이고, Conv는 channel을 바꾼다"**
5. `nn.Conv2d(3, 16, kernel_size=3, padding=1)`의 weight shape는? → **`(16, 3, 3, 3)`**
   - weight shape는 **`(out_channels, in_channels, kernel_height, kernel_width)`** 순서 → `out=16, in=3, K=3` 이므로 `(16, 3, 3, 3)`
   - `(16, 3, 1, 1)`은 `kernel_size=1`(1×1 conv)일 때의 shape. **`padding=1`과 `kernel_size=1`을 헷갈리지 말 것** — padding은 weight shape에 전혀 영향을 주지 않고 **출력 H,W에만** 영향을 줌
   - `nn.Linear`의 `(out_features, in_features)`와 마찬가지로 **출력 쪽이 먼저** 온다고 묶어서 기억
   - 참고: bias shape는 `(16,)`, 총 parameter 수는 `16×3×3×3 + 16 = 448`

**한 줄 정리**: CNN은 이미지를 flatten하지 않고 `(N, C, H, W)` 형태 그대로 받아 `(out_channels, in_channels, kh, kw)` 크기의 작은 filter를 모든 위치에 공유해가며 지역 패턴을 찾는 구조로 — padding과 stride가 `floor((H_in + 2P − K)/S) + 1` 공식대로 공간 크기를 조절하고 MaxPool은 channel과 batch는 그대로 둔 채 H,W만 줄이면서 강한 반응을 남기며, 이렇게 `Conv → BatchNorm2d → ReLU → Pool` block으로 H,W는 줄이고 channel은 늘려간 feature map을 `flatten(start_dim=1)`로 펼쳐 그 flatten_dim과 정확히 같은 `in_features`를 가진 Linear에 연결하면 `(batch_size, num_classes)` logits이 나오고, 이 logits를 softmax 없이 `CrossEntropyLoss`에 그대로 넘기는 것이 CNN classifier의 기본 골격이다.
