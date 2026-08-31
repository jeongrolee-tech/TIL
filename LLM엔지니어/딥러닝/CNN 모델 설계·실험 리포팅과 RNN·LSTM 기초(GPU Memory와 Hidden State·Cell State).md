# 2026-08-31 CNN 모델 설계·실험 리포팅과 RNN·LSTM 기초(GPU Memory와 Hidden State·Cell State)

## 오늘의 TIL 요약

**12장1강. CNN 모델 설계 기준**
- CNN 설계는 layer를 아무렇게나 쌓는 것이 아니라 **"각 layer가 무엇을 하는가"**를 알고 조합하는 일

  | Layer | 역할 | shape에 미치는 영향 |
  | --- | --- | --- |
  | `Conv2d` | 이미지에서 **지역 패턴**을 찾음 | channel을 바꿈, H·W는 padding/stride가 결정 |
  | `BatchNorm2d` | channel별로 분포를 정규화해 **학습을 안정화** | shape 변화 없음 (인자는 channel 수) |
  | `ReLU` | **비선형성**을 추가 | shape 변화 없음 |
  | `MaxPool2d` | 공간 크기를 줄이고 **강한 반응만 남김** | H·W 축소, channel·batch 불변, **학습 parameter 없음** |
  | `Dropout` | 일부 뉴런을 무작위로 꺼서 **과적합 억제** | shape 변화 없음, `eval()`에서 비활성 |
  | `Flatten` → `Linear` | 특징 벡터를 **class별 logits**로 변환 | `(N, C×H×W)` → `(N, num_classes)` |

- ⭐ **설계의 기본 골격**: `Conv → BatchNorm2d → ReLU → Pool`이 한 block이고, 이 block을 반복
  - block을 지날수록 **H·W는 줄이고 channel은 늘린다** (예: `32×32/3ch → 16×16/32ch → 8×8/64ch → 4×4/128ch`)
  - 직관: 초반 layer는 **선·경계 같은 단순한 패턴**을, 후반 layer는 **눈·바퀴 같은 복잡한 패턴**을 담당 → 복잡한 패턴은 종류가 많으므로 channel이 더 필요함
- **설계 기준 5가지**

  | 기준 | 권장 | 이유 |
  | --- | --- | --- |
  | kernel 크기 | **3×3을 깊게** 쌓기 | 3×3을 2번 = 5×5 receptive field인데 parameter는 18 < 25, ReLU도 한 번 더 들어감 |
  | 크기 유지 | `K=3, S=1, P=1` | Conv에서는 H·W를 유지하고 **축소는 Pool에 맡기면** shape 추적이 쉬움 |
  | 채널 증가 | 32 → 64 → 128 처럼 **2배씩** | 공간 정보가 줄어드는 만큼 표현력을 채널로 보상 |
  | 깊이 | 데이터 양에 맞춰 | 32×32 소규모 데이터셋은 block 3개면 충분, 무작정 깊게 쌓으면 과적합 |
  | 정규화 | BatchNorm + Dropout | BatchNorm은 conv 뒤, Dropout은 주로 **classifier 쪽**에 |

- ⛔ **Pooling 횟수 = 크기가 절반이 되는 횟수** — `32 → 16 → 8 → 4 → 2 → 1`. 입력이 32×32인데 pool을 6번 넣으면 feature map이 사라짐
  - "몇 번까지 줄일 수 있는가"의 상한선은 `log2(입력 크기)`
- **parameter 수로 설계를 검증하기** — 설계 직후 반드시 확인하는 습관
  ```python
  import torch
  import torch.nn as nn

  def count_params(model):
      total = sum(p.numel() for p in model.parameters())
      trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
      return total, trainable

  conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
  print(count_params(conv))   # (18496, 18496)  = 64*32*3*3 + 64
  ```
  - Conv parameter = `out_channels × in_channels × kh × kw + out_channels` → **입력 H·W와 무관**
  - Linear parameter = `out_features × in_features + out_features` → **flatten_dim이 크면 여기서 parameter가 폭발**
  - ⭐ 대부분의 CNN에서 **parameter의 절반 이상이 첫 Linear에 몰림** → `AdaptiveAvgPool2d((1,1))`(Global Average Pooling)로 flatten_dim을 channel 수로 고정하면 크게 줄어듦
- **설계 기준을 코드로 정리한 예시**
  ```python
  def conv_block(in_ch, out_ch):
      """Conv -> BN -> ReLU -> Pool 한 덩어리. H,W는 Conv에서 유지하고 Pool에서만 절반으로."""
      return nn.Sequential(
          nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # BN이 뒤에 오면 bias는 불필요
          nn.BatchNorm2d(out_ch),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
      )

  class SimpleCNN(nn.Module):
      def __init__(self, in_channels=3, num_classes=10, base=32):
          super().__init__()
          self.features = nn.Sequential(
              conv_block(in_channels, base),       # (N,3,32,32)  -> (N,32,16,16)
              conv_block(base, base * 2),          # (N,32,16,16) -> (N,64,8,8)
              conv_block(base * 2, base * 4),      # (N,64,8,8)   -> (N,128,4,4)
          )
          self.classifier = nn.Sequential(
              nn.AdaptiveAvgPool2d((1, 1)),        # (N,128,4,4) -> (N,128,1,1)  입력 크기와 무관하게 고정
              nn.Flatten(),                        # (N,128)
              nn.Dropout(0.3),
              nn.Linear(base * 4, num_classes),    # (N,10) logits
          )

      def forward(self, x):
          return self.classifier(self.features(x))

  model = SimpleCNN()
  print(model(torch.randn(8, 3, 32, 32)).shape)   # torch.Size([8, 10])
  ```
  - `bias=False`인 이유: 바로 뒤 `BatchNorm2d`가 평균을 빼면서 bias 효과를 흡수하므로 중복
  - `inplace=True`는 메모리를 아끼는 옵션 (같은 텐서를 다른 곳에서도 쓰는 구조에서는 주의)
- 💡 설계 체크리스트: ① 첫 `Conv2d`의 `in_channels` = 이미지 channel 수, ② 각 block의 `in_channels` = 직전 `out_channels`, ③ pool 횟수를 바꿨으면 **flatten_dim 재계산**, ④ 마지막 `Linear`의 `out_features` = class 수, ⑤ 출력은 **softmax 없는 logits**

**12장2강. CNN 학습 파이프라인 구성**
- 모델만 바뀌었을 뿐, 학습 파이프라인은 **MLP 때와 완전히 동일한 구조**를 재사용

  ```
  seed 고정 → device 설정
    → transform 정의 (train은 augmentation 포함 / valid·test는 순수 변환만)
    → Dataset → train/valid split → DataLoader
    → model → criterion(CrossEntropyLoss) → optimizer(Adam)
    → epoch 반복 { train_one_epoch → evaluate → 기록 → best 갱신 시 checkpoint 저장 }
    → best checkpoint 로드 → 최종 평가 → 리포팅
  ```
- ⭐ **재사용 가능한 함수로 쪼개 두면 MLP·CNN을 그대로 바꿔 끼울 수 있음**
  ```python
  def train_one_epoch(model, loader, criterion, optimizer, device):
      model.train()                                  # Dropout/BatchNorm을 학습 모드로
      total_loss, correct, total = 0.0, 0, 0
      for images, labels in loader:
          images, labels = images.to(device), labels.to(device)

          optimizer.zero_grad()                      # 이전 gradient 초기화 (누적 방지)
          logits = model(images)                     # (N, num_classes)
          loss = criterion(logits, labels)           # softmax 없이 logits 그대로
          loss.backward()
          optimizer.step()

          total_loss += loss.item() * labels.size(0) # .item()으로 float 변환 (텐서를 쌓으면 그래프가 남아 메모리 누수)
          correct += (logits.argmax(1) == labels).sum().item()
          total += labels.size(0)
      return total_loss / total, correct / total

  @torch.no_grad()                                   # 평가에는 gradient가 필요 없음 -> 메모리·속도 이득
  def evaluate(model, loader, criterion, device):
      model.eval()                                   # Dropout off, BatchNorm은 running stats 사용
      total_loss, correct, total = 0.0, 0, 0
      for images, labels in loader:
          images, labels = images.to(device), labels.to(device)
          logits = model(images)
          loss = criterion(logits, labels)
          total_loss += loss.item() * labels.size(0)
          correct += (logits.argmax(1) == labels).sum().item()
          total += labels.size(0)
      return total_loss / total, correct / total
  ```
- **epoch 루프 + best checkpoint**
  ```python
  history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
  best_val_acc, best_epoch = 0.0, -1

  for epoch in range(1, EPOCHS + 1):
      tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
      va_loss, va_acc = evaluate(model, valid_loader, criterion, device)

      for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
          history[k].append(v)

      if va_acc > best_val_acc:                      # 기준 지표를 하나 정해서 그것만 본다
          best_val_acc, best_epoch = va_acc, epoch
          torch.save({"epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "val_acc": va_acc}, "best_cnn.pt")

      print(f"[{epoch:02d}] train {tr_loss:.4f}/{tr_acc:.4f} | valid {va_loss:.4f}/{va_acc:.4f}")
  ```
  - 저장은 **`model` 전체가 아니라 `state_dict`** — 클래스 정의만 있으면 어디서든 복원 가능
- **파이프라인에서 자주 나오는 실수**

  | 실수 | 결과 | 해결 |
  | --- | --- | --- |
  | valid에 augmentation 적용 | 평가가 매번 흔들려 비교 불가 | train transform과 eval transform을 **분리** |
  | 정규화 통계를 전체 데이터로 계산 | **데이터 누수** | mean/std는 **train split에서만** 계산 |
  | `model.eval()` 누락 | Dropout·BatchNorm이 켜진 채 평가 | 평가 함수 첫 줄에 고정 |
  | `optimizer.zero_grad()` 누락 | gradient가 누적되어 학습이 이상해짐 | 매 batch 첫 줄 |
  | `total_loss += loss` (텐서 그대로) | 그래프가 계속 살아 **메모리 폭증** | `loss.item()` 사용 |
  | valid loader에 `shuffle=True` | 순서만 달라질 뿐 무의미, 재현성 저하 | 평가 loader는 `shuffle=False` |

- 💡 `DataLoader(num_workers=N, pin_memory=True)`는 GPU 학습 시 데이터 로딩 병목을 줄여줌 (Windows에서는 `num_workers`를 크게 잡으면 오히려 느려질 수 있어 0~4 범위에서 실험)

**12장3강. GPU Memory 기초**
- GPU memory를 쓰는 주체는 **모델 하나가 아님**

  | 항목 | 설명 | 크기 감각 |
  | --- | --- | --- |
  | 모델 parameter | weight/bias | parameter 수 × 4 byte(float32) |
  | gradient | parameter와 **같은 크기** | parameter와 1:1 |
  | optimizer state | Adam은 momentum·variance **2배** 추가 | SGD보다 훨씬 큼 |
  | **activation** | 역전파를 위해 저장하는 중간 출력 | **batch size에 비례**, 보통 가장 큼 |

  - ⭐ 그래서 OOM의 주범은 대개 모델이 아니라 **batch size × activation**
- **메모리 상태 확인**
  ```python
  print(torch.cuda.memory_allocated() / 1024**2, "MB")   # 살아있는 Tensor가 실제로 쓰는 양
  print(torch.cuda.memory_reserved()  / 1024**2, "MB")   # PyTorch가 OS로부터 미리 확보(예약)해 둔 양
  torch.cuda.empty_cache()                               # 예약분 중 쓰지 않는 부분을 반환
  ```
  - PyTorch는 **caching allocator**를 써서 한 번 받은 메모리를 OS에 바로 돌려주지 않고 재사용함 → `reserved > allocated`가 정상
  - ⭐ **`empty_cache()`의 정확한 의미**: caching allocator가 확보해 둔 **일부 cache memory를 반환하는 데 도움**을 줄 수 있을 뿐, **이미 살아 있는 Tensor의 메모리는 없애지 못함** → 근본적인 부족 해결책이 아니라 **보조 도구**
- ⭐ **`CUDA out of memory`가 났을 때의 대응 순서**
  1. **batch size를 줄인다** ← 가장 먼저, 가장 효과적 (입력과 activation이 동시에 줄어듦)
  2. 이미지 해상도(H, W)를 줄인다
  3. 평가 구간에 `torch.no_grad()` / `torch.inference_mode()`를 적용했는지 확인
  4. 손실·정확도를 **텐서째 누적하고 있지 않은지** 확인 (`loss.item()`)
  5. 모델 크기(channel 수, 깊이)를 줄인다
  6. AMP(`torch.cuda.amp.autocast` + `GradScaler`)로 혼합 정밀도 사용
  7. **gradient accumulation** — 작은 batch를 여러 번 모아 큰 batch 효과를 냄
  8. 필요 없는 변수 `del` 후 `torch.cuda.empty_cache()`
  ```python
  # gradient accumulation: batch 32가 안 들어가면 batch 8을 4번 모아 같은 효과
  accum_steps = 4
  optimizer.zero_grad()
  for i, (images, labels) in enumerate(train_loader):
      loss = criterion(model(images.to(device)), labels.to(device)) / accum_steps
      loss.backward()
      if (i + 1) % accum_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```
- ⛔ 주의할 점
  - **OOM은 학습 시작 직후뿐 아니라 평가 구간에서도 발생** — `no_grad` 없이 valid를 돌리면 activation이 그대로 쌓임
  - Jupyter/Colab에서는 이전 셀의 변수가 살아 있어 메모리를 점유함 → 커널 재시작이 가장 확실
  - 여러 GPU 프로세스가 떠 있으면 `nvidia-smi`로 확인
  - `.to(device)`는 **모델과 데이터 둘 다** 해야 함 (한쪽만 하면 device mismatch 에러)

**12장4강. 필터 수/커널 크기 변경 실험**
- 실험의 원칙: ⭐ **한 번에 하나의 변수만 바꾼다(통제 변인)** — 필터 수와 커널 크기를 동시에 바꾸면 무엇 때문에 좋아졌는지 알 수 없음
  - 나머지는 전부 고정: seed, epoch, optimizer/lr, batch size, transform, split
- **바꿔볼 두 축**

  | 축 | 늘리면 | 줄이면 |
  | --- | --- | --- |
  | **filter 수(`out_channels`, 모델의 너비)** | 표현력↑, 다양한 패턴 탐지, parameter·메모리·시간↑, 데이터가 적으면 과적합 | 가볍고 빠름, 복잡한 패턴은 놓침(과소적합) |
  | **kernel 크기(`kernel_size`, 시야)** | 한 번에 넓은 영역, receptive field↑, parameter는 K²에 비례해 급증 | 세밀한 패턴에 유리, 대신 깊게 쌓아야 넓은 영역을 봄 |

  - ⭐ kernel을 키울 때는 `padding = (K−1)/2`를 함께 바꿔야 H·W가 유지됨 (`K=3→P=1`, `K=5→P=2`, `K=7→P=3`)
- **실험을 쉽게 하려면 모델을 인자로 조립**
  ```python
  class ConfigurableCNN(nn.Module):
      def __init__(self, in_channels=3, num_classes=10, base=32, k=3):
          super().__init__()
          p = k // 2                                   # 홀수 kernel에서 크기 유지용 padding
          chs = [in_channels, base, base * 2, base * 4]
          blocks = []
          for i in range(3):
              blocks += [
                  nn.Conv2d(chs[i], chs[i + 1], kernel_size=k, padding=p, bias=False),
                  nn.BatchNorm2d(chs[i + 1]),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2),
              ]
          self.features = nn.Sequential(*blocks)
          self.classifier = nn.Sequential(
              nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(chs[-1], num_classes)
          )

      def forward(self, x):
          return self.classifier(self.features(x))

  for base in [16, 32, 64]:
      for k in [3, 5]:
          m = ConfigurableCNN(base=base, k=k)
          n = sum(p.numel() for p in m.parameters())
          print(f"base={base:2d}, k={k} -> params={n:,}")
  ```
- **결과 기록 표 형식** (수치는 환경마다 달라지므로 **경향**을 보는 것이 목적)

  | 실험 | base(filter) | kernel | parameter 수 | valid acc | 학습 시간 | 메모 |
  | --- | --- | --- | --- | --- | --- | --- |
  | A(기준) | 32 | 3 | 기준 | 기준 | 기준 | baseline |
  | B | 16 | 3 | ↓↓ | 소폭 ↓ | ↓ | 가벼운데 큰 손해는 없음 |
  | C | 64 | 3 | ↑↑ | 소폭 ↑ 또는 정체 | ↑↑ | train acc만 오르면 과적합 신호 |
  | D | 32 | 5 | ↑ (Conv 기준 약 2.8배) | 비슷 | ↑ | 3×3 두 겹이 더 효율적인 경우가 많음 |

- 💡 해석 포인트
  - **valid acc는 정체하는데 train acc만 오른다 → 과적합** → 모델을 키울 게 아니라 augmentation/Dropout/weight decay를 먼저
  - **train acc도 낮다 → 과소적합** → 이때 filter 수나 깊이를 늘리는 것이 유효
  - **parameter가 2배가 되어도 정확도는 2배가 되지 않음** — 비용 대비 이득(효율)을 함께 보고해야 함

**12장5강. MLP baseline 준비**
- **baseline** = 새 모델이 정말 나은지 판단하기 위한 **비교 기준선**. baseline 없이 "정확도 78%"는 좋은지 나쁜지 알 수 없음
  - 참고선 세 가지: ① 무작위 추측(10 class면 10%), ② 최빈 class 예측, ③ **단순 모델(MLP)**
- ⭐ **공정한 비교를 위해 모델 외의 모든 조건을 고정**: 같은 데이터·split·transform·epoch·optimizer·lr·batch size·seed
  ```python
  class MLPBaseline(nn.Module):
      def __init__(self, in_channels=3, img_size=32, num_classes=10, hidden=512):
          super().__init__()
          in_features = in_channels * img_size * img_size      # 3*32*32 = 3072
          self.net = nn.Sequential(
              nn.Flatten(),                                    # (N,3,32,32) -> (N,3072)  <- 여기서 공간 구조가 사라짐
              nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.3),
              nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.3),
              nn.Linear(hidden // 2, num_classes),             # logits
          )

      def forward(self, x):
          return self.net(x)

  mlp = MLPBaseline()
  print(mlp(torch.randn(8, 3, 32, 32)).shape)   # torch.Size([8, 10])
  ```
- ⭐ **MLP와 CNN 모두 최종 출력이 `(batch_size, num_classes)` logits** → **같은 `CrossEntropyLoss`, 같은 `train_one_epoch`/`evaluate` 함수를 그대로 재사용** 가능. 바뀌는 것은 `model = ...` 한 줄뿐
- ⛔ MLP의 구조적 한계
  - `Flatten`하는 순간 "이 픽셀 옆에 저 픽셀이 있었다"는 **위치 정보가 사라짐**
  - 입력 H·W가 바뀌면 첫 `Linear`의 `in_features`를 전부 다시 계산해야 함
  - 같은 물체가 몇 픽셀만 이동해도 **완전히 다른 입력**으로 취급됨
  - parameter 수는 오히려 CNN보다 많아지기 쉬움 (`3072×512 ≈ 157만`이 첫 층 하나에서 발생)

**12장6강. MLP vs CNN 성능 비교**
- ⭐ 핵심 차이는 **이미지를 어떻게 바라보는가**

  | 항목 | MLP | CNN |
  | --- | --- | --- |
  | 입력 처리 | 이미지를 **flatten**해서 위치 구조를 많이 잃음 | `(N,C,H,W)` 그대로 받아 **작은 영역**을 봄 |
  | 활용하는 정보 | 픽셀 값의 조합 | **지역 패턴 + 공간 구조** |
  | 가중치 | 위치마다 다른 weight | **같은 filter를 모든 위치에 공유** |
  | 위치 이동 | 취약 | 어느 정도 강함 (translation에 둔감) |
  | parameter | 첫 Linear에서 폭증 | 입력 H·W와 무관, 상대적으로 적음 |
  | 출력 | `(batch_size, num_classes)` logits | **동일** |

  - ⭐ 두 모델 모두 `(batch_size, num_classes)` logits을 내므로 **같은 손실 함수·학습 루프를 공유**한다는 점이 자주 나오는 포인트
- **왜 CNN이 이미지에서 더 잘하는가** — parameter가 더 적은데도 이기는 이유는 **inductive bias(모델에 미리 심어 둔 가정)** 때문
  - 가정 ①: 의미 있는 패턴은 **가까운 픽셀끼리** 모여 있다(지역성)
  - 가정 ②: 같은 패턴은 **어디에 있든 같은 패턴**이다(가중치 공유 / translation invariance)
  - 이 두 가정이 이미지에 잘 맞기 때문에, 적은 parameter로도 필요한 것만 효율적으로 학습함
- **비교 리포트 예시** (경향 중심)

  | 지표 | MLP baseline | CNN |
  | --- | --- | --- |
  | parameter 수 | 많음 (첫 Linear에 집중) | 적음 |
  | train acc | 빠르게 상승 | 상승 |
  | **valid acc** | 낮은 수준에서 정체 | **더 높음** |
  | train-valid 격차 | 큼 (과적합 경향) | 상대적으로 작음 |
  | epoch당 시간 | 짧음 | 김 (연산량↑) |

- ⛔ 비교할 때의 함정
  - **epoch·lr·seed가 다르면 비교가 무의미** — 조건을 맞추지 않은 비교로는 결론을 낼 수 없음
  - CNN이 무조건 우월한 것이 아니라 **이미지/공간 데이터에서** 유리한 것. 표 형태(tabular) 데이터에서는 MLP나 트리 계열이 더 나은 경우가 많음
  - 단 한 번의 실행 결과만으로 단정하지 말 것 — seed를 바꾸면 순위가 뒤집힐 정도의 차이일 수도 있음

**12장7강. 실험 결과 리포팅**
- 실험은 **재현 가능하고 남이 읽을 수 있게** 정리해야 비로소 끝남. 리포트의 표준 골격

  | 항목 | 내용 |
  | --- | --- |
  | 1. 실험 목적 | 무엇을 확인하려 했는가 (예: "CNN이 MLP보다 나은지, 얼마나 나은지") |
  | 2. 실험 설정 | 데이터셋·split 비율·transform·모델 구조·optimizer·lr·batch size·epoch·**seed**·하드웨어 |
  | 3. 결과 | 표 + 학습 곡선 그래프 (train/valid loss·accuracy) |
  | 4. 해석 | 숫자가 아니라 **왜 그렇게 나왔는지** |
  | 5. 결론 | 한 문장으로 |
  | 6. 한계·다음 단계 | 무엇을 확인하지 못했는가 |

- **반드시 기록할 지표**: 최종 train/valid loss·accuracy, **best epoch와 그때의 valid 지표**, parameter 수, epoch당 학습 시간, 총 학습 시간
- **학습 곡선 해석 요약**

  | 패턴 | 진단 | 대응 |
  | --- | --- | --- |
  | train↓ valid↓ 함께 감소 | 정상 학습 | 계속 |
  | train↓ **valid↑ 상승 전환** | **과적합** | early stopping, augmentation, Dropout, weight decay |
  | 둘 다 높은 채로 정체 | **과소적합** | 모델 확장, lr 조정, epoch 증가 |
  | 진동이 심함 | lr이 크거나 batch가 작음 | lr↓, batch↑, scheduler |

- ⛔ 리포팅의 정확성
  - **valid로 튜닝하고, 최종 성능은 test로 보고** — valid 최고점을 최종 성능처럼 쓰면 낙관 편향
  - 잘 나온 실행만 고르는 **cherry-picking 금지**. 가능하면 **seed 3개 이상의 평균 ± 표준편차**
  - 실패한 실험도 기록 (무엇이 안 됐는지가 다음 실험의 출발점)
  - 정확도만 쓰지 말 것 — **클래스가 불균형하면 정확도는 착시**를 준다 (필요하면 per-class accuracy, confusion matrix, macro-F1)
- **재현성 정보** — 리포트 맨 아래 고정 블록으로 넣어두면 좋음
  ```python
  import random, numpy as np, torch

  def set_seed(seed=42):
      random.seed(seed); np.random.seed(seed)
      torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True   # 재현성↑ (속도는 약간 손해)
      torch.backends.cudnn.benchmark = False

  print(torch.__version__, torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
  ```
- 💡 결론 문장은 **"무엇을, 어떤 조건에서, 얼마나"**를 담아야 함
  - ❌ "CNN이 더 좋았다"
  - ⭕ "동일한 데이터·전처리·optimizer·seed 조건에서 20 epoch 학습 시, CNN이 MLP baseline보다 valid accuracy가 약 N%p 높았고 parameter 수는 더 적었다. 다만 epoch당 학습 시간은 약 M배 길었다."

**12장8강. CNN 종합 실습 제출**
- 제출 전 체크리스트

  | 구분 | 확인 항목 |
  | --- | --- |
  | 데이터 | train/valid split이 고정되어 있는가, 정규화 통계를 **train에서만** 냈는가, valid에 augmentation이 없는가 |
  | 모델 | 첫 `in_channels`·마지막 `out_features`가 맞는가, flatten_dim이 맞는가, 출력이 **logits**인가 |
  | 학습 | `model.train()`/`model.eval()` 전환, `optimizer.zero_grad()`, 평가에 `no_grad` |
  | 재현성 | seed 고정, best checkpoint 저장, 하이퍼파라미터를 코드 상단 한 곳에 모았는가 |
  | 결과 | 학습 곡선, 비교 표, 해석, 한계까지 포함했는가 |
  | 코드 | 처음부터 끝까지 **한 번에 재실행**해도 에러 없이 돌아가는가 |

- 💡 실습 결과물은 "정확도 몇 %"가 아니라 **"어떤 가설을 어떻게 검증했고 무엇을 배웠는가"**로 정리할 때 가장 가치가 있음

---

**13장1강. Sequence Data와 Hidden State**
- 지금까지 다룬 MLP·CNN은 주로 **하나의 샘플을 독립적으로** 처리했음 (이미지 한 장, 표 데이터 한 행)
- 하지만 실제 데이터 중에는 **순서가 중요한 데이터**가 많음

  | 데이터 | 순서가 중요한 이유 |
  | --- | --- |
  | 문장 | 앞 단어가 뒤 단어의 의미에 영향을 줌 |
  | 주가/매출 | 과거 흐름이 현재 값 해석에 필요 |
  | 센서 데이터 | 시간에 따른 변화 패턴이 중요 |
  | 사용자 행동 로그 | 이전 행동이 다음 행동 예측에 영향 |

  - 이런 데이터를 **시퀀스 데이터(sequence data)**라고 부름
- ⭐ **입력 shape** — PyTorch에서 `batch_first=True`일 때

  ```
  (batch_size, sequence_length, input_size)
  ```

  | 차원 | 의미 | 예시 |
  | --- | --- | --- |
  | `batch_size` | 한 번에 처리하는 샘플 수 | 문장 32개 |
  | `sequence_length` | 각 샘플 안의 시점/토큰 수 | 문장당 단어 10개 |
  | `input_size` | 각 시점이 가진 feature 수 | 단어 embedding 64차원 |

  - ⭐ **중요**: 시퀀스 데이터에서는 첫 번째 차원만 보는 것이 아니라, **샘플 안에 시간축이 하나 더 있다**는 점을 기억해야 함
  - CNN의 `(N, C, H, W)`가 "공간축이 2개"였다면, RNN의 `(B, S, I)`는 **"시간축이 1개"**
- **RNN(순환 신경망, Recurrent Neural Network)** = 이전 내용(**hidden state**)을 기억하면서 다음 데이터를 처리하는 신경망
  - ⭐ **hidden state** = "지금까지 읽은 내용을 요약한 **메모장**". 현재 입력 `x_t`와 이전 hidden state `h_{t-1}`을 함께 사용해 새 hidden state `h_t`를 만듦

  ```
  h_t = tanh( W_xh · x_t  +  W_hh · h_{t-1}  +  b )

  x1 ──┐      x2 ──┐      x3 ──┐
       ▼           ▼           ▼
  h0 ─► h1 ──────► h2 ──────► h3      (같은 W를 매 시점 반복 사용)
  ```

  - ⭐ **모든 시점이 같은 weight(`W_xh`, `W_hh`)를 공유** — CNN의 가중치 공유가 "공간축"이었다면 RNN은 **"시간축"**에서의 공유. 덕분에 시퀀스 길이가 달라져도 같은 모델을 쓸 수 있음
  - `h_0`을 넘기지 않으면 PyTorch가 **0으로 자동 초기화**
- **시퀀스 문제의 유형**

  | 유형 | 사용하는 출력 | 예시 |
  | --- | --- | --- |
  | many-to-one | **마지막 hidden state**만 | 문장 감성 분류, 시계열 다음 값 예측 |
  | many-to-many (동일 길이) | **모든 시점의 output** | 품사 태깅, 개체명 인식 |
  | one-to-many / seq2seq | 인코더-디코더 | 번역, 요약 |

- ⛔ 실제 데이터는 문장 길이가 제각각 → **padding**으로 길이를 맞추고, 패딩 부분이 학습에 끼어들지 않게 `pack_padded_sequence`나 mask를 사용하는 것이 정석
- 하지만 RNN에는 문제가 있음: **문장이 길어질수록 처음 정보를 점점 잊어버림** → 이를 **장기 의존성(Long-term Dependency) 문제**라고 함

**13장2강. RNN forward 흐름과 출력 shape**
- PyTorch의 `nn.RNN`, `nn.LSTM`, `nn.GRU`는 보통 **`output`과 `h_n`을 함께 반환**

  | 값 | 의미 | shape | 예시 |
  | --- | --- | --- | --- |
  | `output` | **모든 시점(time step)**의 hidden state | `(batch, seq_len, hidden)` | `(4, 6, 8)` |
  | `h_n` | **마지막 시점**의 hidden state | `(num_layers × 방향, batch, hidden)` | `(1, 4, 8)` |

  - Batch = 4, Sequence 길이 = 6, Hidden size = 8 인 경우
  ```
  output (4, 6, 8)                       h_n (1, 4, 8)
  batch1 : h1 h2 h3 h4 h5 h6             batch1 : h6
  batch2 : h1 h2 h3 h4 h5 h6      vs     batch2 : h6
  batch3 : h1 h2 h3 h4 h5 h6             batch3 : h6
  batch4 : h1 h2 h3 h4 h5 h6             batch4 : h6
  ```
  즉 `output`은 6개 시점의 hidden state를 **모두 저장**하고, `h_n`은 **마지막 시점(h6)만** 가지고 있음
- ⭐ **왜 `h_n`의 첫 차원이 1일까?**
  - 첫 번째 차원은 **batch가 아니라 `num_layers × 방향 수`**

  | 구성 | `h_n` shape |
  | --- | --- |
  | layer=1, 단방향 | `(1, batch, hidden)` |
  | layer=2, 단방향 | `(2, batch, hidden)` |
  | layer=1, 양방향 | `(2, batch, hidden)` |
  | layer=2, 양방향 | `(4, batch, hidden)` |

  - ⛔ `batch_first=True`는 **`output`에만 적용**되고 **`h_n`은 언제나 `(layer×방향, batch, hidden)`** → 이 비대칭이 shape 혼동의 최대 원인
- **기본 forward**
  ```python
  import torch
  import torch.nn as nn

  x = torch.randn(4, 6, 3)
  # 4 : 한 번에 처리하는 데이터 개수(batch)
  # 6 : 각 데이터의 시점(time step) 개수
  # 3 : 한 시점의 입력 특징(feature) 개수
  # x == [B, S, I]  ⭐ (B,T,F) == [batch, sequence_length, input_size]

  rnn = nn.RNN(input_size=3, hidden_size=5, batch_first=True)
  # input_size=3     : 한 시점에 입력되는 feature 개수
  # hidden_size=5    : RNN이 기억하는 정보(hidden state)의 크기 -> 매 시점 hidden state 크기가 5
  # batch_first=True : 입력을 (batch, sequence_length, input_size) 형태로 받겠다는 의미

  output, h_n = rnn(x)
  print('output:', output.shape)   # ⭐ (B, S, H)   -> torch.Size([4, 6, 5])
  print('h_n:',    h_n.shape)      # ⭐ (L×D, B, H) -> torch.Size([1, 4, 5])
  ```
  - ⭐ 단방향 1-layer라면 **`output[:, -1, :]`와 `h_n[-1]`은 같은 값** (양방향이면 성립하지 않음)
    ```python
    print(torch.allclose(output[:, -1, :], h_n[-1]))   # True
    ```
- **`batch_first=False`(기본값)인 RNN은 `[seq_len, batch, input_size]` 형태를 받음**
  ```python
  x_batch_first = torch.randn(3, 5, 4)                  # [batch, seq, input]
  rnn = nn.RNN(input_size=4, hidden_size=6, batch_first=False)

  x_time_first = x_batch_first.permute(1, 0, 2)         # [seq, batch, input]
  out, h = rnn(x_time_first)
  print(out.shape, h.shape)   # torch.Size([5, 3, 6]) torch.Size([1, 3, 6])
  ```
  - ⛔ `batch_first`를 잘못 두면 **에러 없이 조용히 잘못 학습됨** — batch를 시간축으로 착각해도 shape이 맞아버리는 경우가 있기 때문. 코드 전체에서 `batch_first=True`로 통일하는 편이 안전
- **다층·양방향 옵션**
  ```python
  rnn = nn.RNN(input_size=3, hidden_size=5, num_layers=2,
               bidirectional=True, batch_first=True, dropout=0.2)
  output, h_n = rnn(torch.randn(4, 6, 3))
  print(output.shape)   # torch.Size([4, 6, 10])  <- hidden_size * 2(양방향)
  print(h_n.shape)      # torch.Size([4, 4, 5])   <- (2 layers × 2 dir, batch, hidden)
  ```
- **분류기로 연결하기 (many-to-one)**
  ```python
  class RNNClassifier(nn.Module):
      def __init__(self, input_size=3, hidden_size=64, num_classes=2):
          super().__init__()
          self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
          self.fc = nn.Linear(hidden_size, num_classes)

      def forward(self, x):                 # x: (B, S, I)
          output, h_n = self.rnn(x)
          last = h_n[-1]                    # (B, H)  마지막 layer의 마지막 시점
          return self.fc(last)              # (B, num_classes) logits

  print(RNNClassifier()(torch.randn(4, 6, 3)).shape)   # torch.Size([4, 2])
  ```
  - ⛔ `nn.Linear`에 `output`을 그대로 넣으면 `(B, S, num_classes)`가 나옴 → **many-to-one이면 마지막 시점만 골라야 함**

**13장3강. 장기 의존성과 경사 소실**
- RNN 학습은 **BPTT(Backpropagation Through Time)** — 시간축을 따라 펼친 뒤 마지막 시점에서 처음 방향으로 gradient를 되돌려 보냄
- ⭐ 이때 gradient는 시점을 거슬러 갈 때마다 **같은 `W_hh`가 반복해서 곱해짐**

  ```
  ∂L/∂h_1  ∝  W_hh^(T-1) × (활성함수 미분들의 곱)
  ```

  | 반복 곱셈의 결과 | 현상 | 증상 |
  | --- | --- | --- |
  | 값이 1보다 작음 | **경사 소실(vanishing gradient)** | 앞쪽 시점의 weight가 거의 갱신되지 않음 → **오래된 정보를 못 배움** |
  | 값이 1보다 큼 | **경사 폭발(exploding gradient)** | loss가 `NaN`/`inf`로 튐, 학습이 발산 |

  - `tanh`의 미분은 최대 1(대부분 그보다 작음) → 구조적으로 **소실 쪽이 훨씬 흔함**
  - 그래서 **문장이 길어질수록 처음 정보를 점점 잊어버리는 장기 의존성 문제**가 생김
    - 예: "나는 프랑스에서 자랐고 … (긴 문장) … 그래서 나는 **프랑스어**를 잘한다" — 정답 단서가 너무 앞에 있으면 RNN이 놓침
- **대응 방법**

  | 문제 | 해결책 | 설명 |
  | --- | --- | --- |
  | 경사 **폭발** | **gradient clipping** | gradient norm이 임계값을 넘으면 잘라냄 (즉효, 사실상 필수) |
  | 경사 **소실** | **LSTM / GRU** | gate + cell state로 gradient가 흐를 **덧셈 경로**를 만듦 |
  | 〃 | 양방향·다층 구성, 적절한 초기화 | 정보 경로를 늘림 |
  | 〃 | **Attention / Transformer** | 시점 간 **직접 연결** → 거리와 무관하게 정보 접근 |

  ```python
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # step 직전에!
  optimizer.step()
  ```
  - ⛔ clipping은 반드시 **`backward()` 이후, `step()` 이전**에 넣어야 의미가 있음
- 💡 정리하면 — **폭발은 clipping으로 막고, 소실은 구조(LSTM/GRU)로 푼다**

**13장4강. LSTM Gate와 Cell State**
- **LSTM (Long Short-Term Memory)** = RNN의 기억력 부족을 개선한 모델
  - ⭐ 핵심 아이디어: **"중요한 정보는 오래 기억하고, 필요 없는 정보는 버린다"** — 즉 **무엇을 기억하고 무엇을 잊을지 스스로 결정**함
- **Cell State (셀 상태)** = LSTM의 **장기 기억 저장소**
  - RNN은 hidden state 하나만 계속 전달해서 오래된 정보를 잘 잊어버렸지만, LSTM은 **Cell State라는 별도의 통로**를 만들어 중요한 정보를 오래 보관
  ```
  단어1 → 단어2 → 단어3 → 단어4
     ==============================>
              Cell State
  ```
  - Cell State는 문장을 끝까지 흐르면서 중요한 정보를 전달함
  - ⭐ 이 통로의 갱신이 **곱셈이 아니라 덧셈(`C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t`)**이라는 점이 핵심 — 반복 곱셈이 줄어들어 **gradient가 오래 살아남음**(= 경사 소실 완화)
- **Gate** = Cell State를 관리하는 **문지기**. 세 가지가 있음

  | Gate | 결정하는 것 | 수식(개념) | 직관 |
  | --- | --- | --- | --- |
  | **Forget Gate** `f_t` | 무엇을 **잊을지** | `σ(W_f·[h_{t-1}, x_t] + b_f)` | 이전 기억 중 버릴 비율 |
  | **Input Gate** `i_t` | 무엇을 **기억할지** | `σ(W_i·[h_{t-1}, x_t] + b_i)` | 새 정보 중 담을 비율 |
  | **Output Gate** `o_t` | 무엇을 **출력할지** | `σ(W_o·[h_{t-1}, x_t] + b_o)` | 기억 중 지금 꺼내 쓸 부분 |

  ```
  C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ tanh(W_c·[h_{t-1}, x_t] + b_c)   # 장기 기억 갱신
  h_t = o_t ⊙ tanh(C_t)                                          # 밖으로 내보내는 값
  ```
  - ⭐ gate는 모두 **sigmoid** → 출력이 `0~1` → **"몇 %를 통과시킬지"를 나타내는 비율**로 작동 (0이면 완전 차단, 1이면 완전 통과)
  - gate의 weight도 **학습으로 결정됨** — 사람이 규칙을 정해주는 것이 아님
- **PyTorch에서의 LSTM — 반환값이 하나 더 많음**
  ```python
  lstm = nn.LSTM(input_size=3, hidden_size=8, batch_first=True)
  x = torch.randn(4, 6, 3)

  output, (h_n, c_n) = lstm(x)      # ⭐ RNN과 달리 (h_n, c_n) 튜플
  print(output.shape)   # torch.Size([4, 6, 8])   (B, S, H)   모든 시점의 hidden state
  print(h_n.shape)      # torch.Size([1, 4, 8])   (L×D, B, H) 마지막 hidden state
  print(c_n.shape)      # torch.Size([1, 4, 8])   (L×D, B, H) 마지막 cell state
  ```
  - ⛔ `output, h_n = lstm(x)`로 받으면 `h_n`에 **튜플이 통째로** 들어가 뒤에서 이상한 에러가 남 → 반드시 `output, (h_n, c_n)` 형태로 언패킹
  - `output`에는 hidden state만 담기고 **cell state는 담기지 않음** (cell state는 내부 기억 통로)
- **RNN / LSTM / GRU 비교**

  | 모델 | 전달하는 상태 | gate 수 | parameter | 특징 |
  | --- | --- | --- | --- | --- |
  | `nn.RNN` | `h` | 0 | 가장 적음 | 단순하지만 장기 의존성에 약함 |
  | `nn.LSTM` | `h`, `c` | 3 (forget/input/output) | 가장 많음(≈RNN의 4배) | 장기 기억에 강함, 기본 선택 |
  | `nn.GRU` | `h` | 2 (reset/update) | 중간(≈RNN의 3배) | LSTM과 비슷한 성능에 더 가볍고 빠름 |

  - 셋 다 **인터페이스가 거의 같아** 코드에서 클래스 이름만 바꿔 실험할 수 있음 (LSTM만 반환값이 `(h_n, c_n)`)
- 💡 LSTM은 "기억을 무조건 오래 붙잡는 모델"이 아니라 **잊는 것까지 학습하는 모델**이라고 이해하는 편이 정확함 (Forget Gate가 없으면 오래된 잡음까지 쌓임)

**오늘 배운 것 한눈에 정리**
- CNN 설계는 **`Conv → BN → ReLU → Pool` block을 반복하며 H·W는 줄이고 channel은 늘리는 것**이 기본, 3×3을 깊게 쌓는 편이 효율적
- 학습 파이프라인은 **MLP와 동일** — 모델만 갈아 끼우면 되고, 출력이 둘 다 `(batch_size, num_classes)` logits이라 손실 함수·학습 함수를 공유
- GPU memory는 **parameter + gradient + optimizer state + activation**이 함께 쓰며, OOM이면 **batch size를 가장 먼저** 줄임. `empty_cache()`는 보조 도구일 뿐
- 실험은 **한 번에 하나의 변수만** 바꾸고, 나머지 조건과 seed를 고정한 뒤 표·곡선·해석·한계까지 갖춰 리포팅
- RNN은 **hidden state**로 순서 정보를 다음 시점에 전달하지만, 반복 곱셈 때문에 **경사 소실 → 장기 의존성 문제**가 생김
- LSTM은 **Cell State + 3개의 Gate(Forget/Input/Output)**로 무엇을 잊고 기억하고 출력할지 스스로 학습해 이 문제를 완화

**데일리 퀴즈 정리**
1. `torch.cuda.empty_cache()`에 대한 설명 → **PyTorch가 예약해 둔 일부 cache memory를 반환하는 데 도움을 줄 수 있지만, 살아 있는 Tensor의 메모리는 없애지 못한다**
   - PyTorch는 **caching allocator**로 한 번 확보한 GPU memory를 재사용함 → `empty_cache()`는 그중 **사용되지 않는 예약분**을 되돌릴 뿐
   - ⛔ **이미 살아 있는 Tensor가 차지하는 memory는 없애주지 않음** → 근본적인 memory 부족 해결책이 아니라 **보조 도구**
   - 실제로 줄이려면 `del`로 참조를 끊거나 스코프를 벗어나게 해야 하고, 그 다음에야 `empty_cache()`가 의미를 가짐
2. GPU memory가 부족해 `CUDA out of memory` 오류가 발생했을 때 **가장 먼저 줄여볼 것** → **batch size**
   - batch size를 줄이면 한 번에 처리하는 **입력과 activation 크기가 함께 줄어** memory 사용량이 감소
   - activation은 batch size에 거의 비례하고 보통 memory 사용량의 가장 큰 비중을 차지하므로 효과가 가장 즉각적
   - 그 다음 순서: 이미지 해상도 ↓ → 평가에 `no_grad` 적용 → 모델 크기 ↓ → AMP → gradient accumulation
3. MLP와 CNN의 이미지 처리 방식 차이 → **MLP는 이미지를 flatten해서 위치 구조를 많이 잃지만, CNN은 작은 영역을 보며 지역 패턴과 공간 구조를 활용한다**
   - ⭐ 함정 선지: "그래서 손실 함수나 학습 함수가 달라진다"는 **틀림** — 두 모델 모두 `(batch_size, num_classes)` 형태의 **logits**를 출력하므로 **같은 `CrossEntropyLoss`와 같은 학습 함수를 그대로 사용**할 수 있음
   - 바뀌는 것은 모델 내부 구조뿐이고 파이프라인은 재사용된다는 점이 핵심
4. RNN이 현재 입력과 함께 사용하여, 지금까지 읽은 정보를 요약해 다음 시점으로 전달하는 값 → **hidden state**
   - **"지금까지 읽은 내용을 요약한 메모장"**에 비유 — 현재 입력 `x_t`와 이전 hidden state `h_{t-1}`을 사용해 새로운 `h_t`를 만듦 (`h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b)`)
   - 시퀀스의 **순서 정보를 다음 시점으로 전달하는 핵심 요소**이며, 모든 시점이 같은 weight를 공유함
   - 참고: LSTM에서는 여기에 **cell state(장기 기억 통로)**가 추가됨
5. 다음 코드에서 `output`의 shape는?
   ```python
   x = torch.randn(4, 6, 3)
   rnn = nn.RNN(input_size=3, hidden_size=8, batch_first=True)
   output, h_n = rnn(x)
   print(output.shape)
   ```
   → **`torch.Size([4, 6, 8])`**
   - `batch_first=True`이므로 입력 shape는 `(batch_size, seq_len, input_size) = (4, 6, 3)`
   - `output`에는 **모든 시점의 hidden state**가 담기므로 `(batch_size, seq_len, hidden_size) = (4, 6, 8)`
   - ⛔ `(1, 4, 8)`은 **`h_n`의 shape** — `(num_layers × 방향, batch, hidden)`이며 마지막 시점만 담고 있음. `output`과 혼동하지 말 것

**한 줄 정리**: CNN 설계는 `Conv → BatchNorm → ReLU → Pool` block을 반복하며 H·W는 줄이고 channel은 늘리는 것이 기본이고 — 학습 파이프라인은 MLP와 완전히 같아서 두 모델 모두 `(batch_size, num_classes)` logits을 내므로 같은 `CrossEntropyLoss`와 학습 함수를 공유하며 같은 seed·epoch·optimizer 조건에서 baseline 대비 성능을 리포팅해야 하고, GPU memory는 parameter·gradient·optimizer state·activation이 함께 쓰기 때문에 OOM이 나면 activation에 직결되는 **batch size를 가장 먼저** 줄이는 것이 정석(`empty_cache()`는 예약분만 돌려주는 보조 도구), 그리고 순서가 중요한 데이터는 `(batch, seq_len, input_size)` 형태로 RNN에 넣어 hidden state로 과거를 요약해 전달하는데 `W_hh`가 반복 곱해지며 경사가 소실되어 장기 의존성 문제가 생기므로 LSTM이 **Cell State라는 덧셈 통로와 Forget·Input·Output 세 Gate**로 무엇을 잊고 기억하고 출력할지 스스로 학습해 이를 완화한다.
