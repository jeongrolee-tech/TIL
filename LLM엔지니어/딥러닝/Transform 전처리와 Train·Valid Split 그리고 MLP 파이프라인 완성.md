# 2026-08-26 Transform 전처리와 Train·Valid Split 그리고 MLP 파이프라인 완성

## 오늘의 TIL 요약

**7장3강. Transform과 전처리 흐름**
- transform: 데이터 하나를 모델이 학습하기 좋은 형태로 바꾸는 함수(또는 함수 묶음) — TorchVision에서는 이미지/Tensor에 적용
- 대표 transform: `Resize`(크기 맞춤), `ToTensor`(PyTorch Tensor로 변환), `Normalize`(채널별 평균/표준편차로 분포 조정), `RandomHorizontalFlip`(학습 데이터 증강용 랜덤 flip)
- `transforms.Compose([...])`로 여러 transform을 순서대로 연결해 하나의 전처리 파이프라인처럼 사용
  - 순서 주의: `Resize`/`RandomCrop` 같은 이미지 단계 transform은 먼저, `ToTensor`는 그다음, `Normalize`는 Tensor 값에 적용되므로 항상 `ToTensor` 뒤에 위치
- 이미지 전처리 5단계: 파일 읽기 → 크기 맞추기(Resize) → Tensor 변환(ToTensor) → 값 분포 정규화(Normalize) → batch로 묶기(DataLoader)
- `Normalize(mean, std)`: `정규화된 값 = (원래 값 - mean) / std` — 예: `[0, 1]` 범위 값에 `mean=0.5, std=0.5`를 적용하면 대략 `[-1, 1]` 범위로 이동 (`x_norm = (x - 0.5) / 0.5`)
  - 정규화는 모델이 특정 스케일에 과하게 민감해지는 것을 줄이고 학습을 안정화하는 데 도움
  - `mean=std=0.5`는 간단한 placeholder이며, ImageNet으로 사전학습된 모델을 쓸 때는 보통 `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`처럼 그 모델이 학습된 통계를 그대로 맞춰줘야 함. 처음부터 직접 학습하는 경우엔 train set에서 계산한 채널별 평균/표준편차를 사용하는 것이 더 정확
- ⛔ valid transform에 랜덤 augmentation(`RandomHorizontalFlip` 등)을 넣으면 평가 결과가 실행할 때마다 흔들림 → 랜덤 요소는 train transform에만 포함
- ⛔ `random_split`이 만든 Subset들은 같은 원본 Dataset을 공유 → 원본 Dataset의 `transform` 속성 하나만 바꾸면 train/valid에 서로 다른 transform이 적용되지 않음 (자세한 해결 패턴은 7장4강 심화 참고)

```python
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),  # 랜덤은 train에서만
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

valid_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
```

**7장4강. Batch size, Shuffle, Train/Valid/Test Split**
- 데이터 역할 분리: train(파라미터 학습) / valid(설정 비교·best 모델 선택) / test(모든 선택이 끝난 뒤 최종 성능을 한 번만 확인)
- loader 설정 관례: train은 `shuffle=True`, valid/test는 `shuffle=False`
- shuffle이 필요한 이유: 데이터가 특정 순서(예: class 0이 앞, class 1이 뒤)로 정렬돼 있으면 batch마다 class 분포가 심하게 달라져 학습이 그 순서에 영향을 받을 수 있음
- `random_split` vs `train_test_split`
  | 구분 | train_test_split | random_split |
  | --- | --- | --- |
  | 대상 | 리스트/NumPy/Pandas 등 일반 데이터 | PyTorch Dataset |
  | 동작 | 데이터를 직접 나눠 반환 | Subset(원본 Dataset을 참조) 반환 |
  | 특징 | X, y 등 여러 배열을 같은 기준으로 동시 분할 | 하나의 Dataset만 분할(Dataset이 이미 `(image, label)`처럼 데이터+라벨을 함께 가짐) |
- 마지막 batch는 `batch_size`보다 작을 수 있음. valid/test는 모든 샘플을 평가해야 하므로 보통 `drop_last=False` 유지(train에서는 BatchNorm 등으로 마지막 불균일 batch를 피하고 싶을 때 `drop_last=True`를 쓰기도 함)
- 재현 가능한 split이 필요하면 seed가 고정된 `torch.Generator`를 `random_split`의 `generator` 인자로 전달
  ```python
  generator = torch.Generator().manual_seed(42)
  train_part, valid_part = random_split(dataset, [0.8, 0.2], generator=generator)
  ```
- (심화) 단순 무작위 분할이 항상 안전하지 않은 경우
  - class 비율이 매우 불균형 → stratified split으로 각 split의 class 비율을 맞춤
  - 같은 사람·환자·장비에서 나온 샘플이 여러 개 → group split으로 같은 group이 서로 다른 split에 섞이지 않게 함
  - 시간 순서가 있는 데이터 → 미래 정보가 과거 학습에 들어가지 않도록 time-based split
  - 전처리 통계(평균/표준편차, vocabulary 등)는 반드시 train set에서만 계산해 valid/test에 적용 — split 이전에 전체 데이터로 계산하면 data leakage 발생
- 자주 하는 실수: valid loader에 `shuffle=True`를 사용 — 틀린 것은 아니지만 재현 가능한 평가를 위해 보통 `shuffle=False`를 사용

**7장4강 심화. random_split과 SubsetWithTransform으로 split별 transform 관리하기**
- 핵심 오해 포인트: `random_split()`은 데이터를 복사해서 새 Dataset을 만드는 함수가 아니라, 원본 Dataset은 하나 그대로 두고 **사용할 index만 서로 다르게 나눠 갖는** 함수
  - 예: 원본이 `[0..9]`일 때 Train은 `[0,2,5,7,8]` index, Valid는 `[1,3,4,6,9]` index만 사용하고, 둘 다 같은 원본 Dataset 객체를 참조
- 그래서 원본 Dataset의 `transform` 속성을 계속 바꿔치기하는 방식으로는 train/valid의 transform을 독립적으로 관리할 수 없음: `dataset.transform = train_transform` 후 `dataset.transform = valid_transform`으로 덮어쓰면 Train도 Valid도 결국 마지막에 설정한 transform을 함께 사용하게 됨
- 해결 패턴: 원본 Dataset(transform 없음)을 먼저 index로만 분할한 뒤, 각 split의 index와 원하는 transform을 함께 묶어주는 `SubsetWithTransform` 같은 Wrapper Dataset을 사용
  ```python
  class SubsetWithTransform(Dataset):
      def __init__(self, base_dataset, indices, transform):
          self.base_dataset = base_dataset
          self.indices = indices
          self.transform = transform

      def __len__(self):
          return len(self.indices)

      def __getitem__(self, i):
          image, label = self.base_dataset[self.indices[i]]
          return self.transform(image), label

  train_dataset = SubsetWithTransform(raw_dataset, train_part.indices, train_transform)
  valid_dataset = SubsetWithTransform(raw_dataset, valid_part.indices, valid_transform)
  ```
- 최종 구조: 하나의 `raw_dataset` 아래로 Train index + `train_transform`, Valid index + `valid_transform`이 각각 독립적인 Wrapper Dataset으로 분리됨
- 기억할 3가지: ① `random_split()`은 데이터 복사가 아니라 index 분할 ② Train/Valid는 서로 다른 index를 쓰지만 같은 원본 Dataset 객체를 참조 ③ split마다 다른 transform이 필요하면 Wrapper가 index와 transform을 함께 관리
- 이 문제 자체는 data leakage가 아님 — Wrapper를 쓰는 이유는 train/valid의 transform을 독립적으로 관리하기 위해서. data leakage는 예를 들어 train+valid+test 전체로 평균/표준편차를 계산해 그 값을 학습 전처리에 쓰는 것처럼, valid/test의 정보가 학습 과정에 새어 들어갈 때 발생
- 비유: 원본 Dataset이 하나의 데이터 창고라면, `random_split()`은 Train/Valid가 사용할 번호표를 나누는 것이고, `SubsetWithTransform`은 각 번호표에 맞는 전처리 방법까지 따로 붙여주는 역할

**7장5강. 데이터 파이프라인 디버깅**
- 학습 전에는 항상 `logits shape`, `target shape`, `target dtype`을 점검
  | 문제 유형 | loss 함수 | 모델 출력 shape | target shape | target dtype |
  | --- | --- | --- | --- | --- |
  | 회귀 | `MSELoss` | `(N, 1)` 또는 `(N, D)` | 출력과 같음 | float |
  | 이진 분류 | `BCEWithLogitsLoss` | `(N, 1)` 또는 `(N,)` | 출력과 같음 | float |
  | 다중 분류 | `CrossEntropyLoss` | `(N, C)` | `(N,)` | long |

**7장 전체 정리**
- Dataset은 샘플/label을 어떻게 꺼낼지, DataLoader는 그걸 batch로 묶어 학습 루프에 공급
- TensorDataset은 이미 준비된 Tensor를 묶을 때, Custom Dataset은 파일 로딩·전처리·label 생성 같은 복잡한 로직이 필요할 때 사용
- transform은 원본 데이터를 모델 입력에 맞게 바꾸는 전처리 파이프라인이며, train에는 augmentation을 넣고 valid는 보통 랜덤 변환 없이 고정
- train/valid split과 loader 구성으로 학습과 검증 흐름을 분리하고, split마다 다른 transform이 필요하면 Wrapper Dataset으로 index와 transform을 함께 관리
- 학습 전에는 첫 batch의 shape, dtype, label, device, forward, loss 계산을 반드시 확인

**8장1강. nn.Module 구조와 forward 설계**
- `__init__`은 layer를 준비하는 공간, `forward`는 입력 Tensor가 layer를 통과하는 순서를 정의하는 공간
- `model(x)`를 호출하면 내부적으로 `forward(x)`가 실행됨 — 정확히는 `nn.Module.__call__`이 먼저 실행되며, 그 안에서 등록된 forward hook 등을 처리한 뒤 `forward()`를 호출하므로 관례상 `model.forward(x)`를 직접 부르지 않고 `model(x)`로 호출
- 실수: MLP 입력에서 batch 차원을 함께 없애버리는 것 → `torch.flatten(x, start_dim=1)`처럼 batch 차원(0번)은 유지하고 1번 차원부터 펼쳐야 함
- 학습 시작 전에는 더미 입력으로 `logits` shape를 반드시 확인

**8장2강. MLP 모델 클래스 완성**
- 28x28 흑백 이미지는 flatten 후 784개 feature가 됨 → `MLP: Flatten → Linear → ReLU → Linear`, 마지막 출력 shape는 `(batch_size, num_classes)`
- parameter 수 계산: `Linear(in, out)`의 parameter 수는 `weight(in*out) + bias(out)`
  - 예: `fc1(784→128)` = `784*128 + 128 = 100,352 + 128`, `fc2(128→10)` = `128*10 + 10 = 1,280 + 10` → 전체 `101,770`개
  - parameter 수는 학습 시간, 메모리 사용량, 과적합 가능성과 직결되므로 모델 크기를 키우기 전에 먼저 계산해보는 습관이 유용
- 자주 하는 실수
  - `input_dim`에 이미지 한 변 길이(28)만 넣기 → `input_dim=1*28*28`(=784)로 flatten된 크기를 넣어야 함
  - 출력 `num_classes`를 실제 label 종류 수와 다르게 설정 → class가 10개면 `num_classes=10`

**8장3강~7강 복습 (loss/optimizer 연결 ~ epoch 로그·시각화)**
- 이전에 배운 내용의 재확인이라 세부 내용은 생략하되, 8장8강 종합 실습에서 다시 조립되는 구성 요소만 정리하면: loss/optimizer 연결 → train loop(`zero_grad→forward→loss→backward→step`) → validation loop(`model.eval()+no_grad`, 파라미터 업데이트 없음) → accuracy를 배치 평균이 아니라 전체 샘플 수 기준으로 누적 → epoch 단위 history 저장과 loss/acc 곡선 시각화

**8장8강. MLP 종합 실습**
- 전체 파이프라인: 데이터 생성 → TensorDataset → train/valid split → DataLoader → MLP 모델 → loss/optimizer → train loop → validation loop → history 저장 → 그래프 시각화
  | 단계 | 역할 |
  | --- | --- |
  | 데이터 준비 | 입력 Tensor와 정답 Tensor 생성 |
  | Dataset 구성 | 입력과 정답을 하나의 데이터셋으로 묶기 |
  | DataLoader 구성 | batch 단위로 데이터를 공급 |
  | 모델 정의 | MLP 구조 만들기 |
  | 학습 준비 | loss와 optimizer 설정 |
  | 학습/검증 | epoch 단위로 train과 validation 실행 |
  | 결과 확인 | 로그 출력과 그래프 시각화 |
- 실습 조건: seed 42, 700/150/150 분할, `4 → 32 → 3` MLP, Adam(`lr=0.01`), 20 epoch, CPU. 마지막(20epoch) 값은 `train_loss=0.0399, train_acc=0.9914, valid_loss=0.0673, valid_acc=0.9667`이었고, valid loss가 가장 낮았던 16epoch의 `best_state`를 복원한 모델로 test set을 한 번만 평가해 `final test=1.000`을 얻음 — 하드웨어/버전에 따라 소수점은 달라질 수 있으므로 절대값 자체를 암기하지 않기
  - `best_state` 패턴: 매 epoch마다 valid loss가 갱신되면 `model.state_dict()`를 복사해 저장해두고, 학습이 끝난 뒤 그 시점의 가중치로 복원(load) 후 test 평가 — 이후 배울 early stopping도 같은 아이디어(valid 성능이 더 이상 좋아지지 않으면 학습을 멈추고 best 시점으로 복원)를 학습 종료 시점에도 적용한 것
- 그래프는 train/valid 곡선의 방향과 간격을 함께 읽음 — train만 계속 좋아지고 valid loss가 다시 오르면 과적합 의심
- ⛔ test 결과를 본 뒤 설정을 다시 바꾸면 test가 사실상 또 하나의 validation이 되어버림(정보가 새어 들어가는 것과 같은 문제) → 최종 선택이 끝난 뒤 test는 한 번만 확인

**8장 전체 정리**
- `nn.Module`로 모델 구조를 정의하고, loss/optimizer를 모델 학습에 연결
- train loop와 validation loop를 명확히 구분(파라미터 업데이트 여부가 핵심 차이)
- accuracy 등 metric은 전체 샘플 기준으로 누적해서 계산
- epoch 단위 history를 저장·시각화하고, 최종적으로 전체 MLP 학습 파이프라인을 하나의 코드로 연결

**데일리 퀴즈 정리 (정답 포함)**
1. PyTorch에서 CNN 모델이 일반적으로 기대하는 이미지 batch의 shape 순서는? → **정답: `(batch_size, channels, height, width)`**. PyTorch 이미지 Tensor는 `(C, H, W)`이며 batch가 되면 `(N, C, H, W)`. `(batch_size, height, width, channels)`(TensorFlow/NumPy 이미지 배열에서 흔한 형태)와 착각하지 않기
2. valid 데이터에는 보통 랜덤 augmentation을 적용하지 않는 이유는? → **정답: 재평가할 때마다 결과의 일관성을 잃을 수 있기 때문**. valid에 랜덤 변환을 넣으면 같은 모델이라도 실행할 때마다 평가 점수가 흔들려 설정 비교의 기준으로 쓰기 어려워짐
3. 다중 분류에서 `CrossEntropyLoss`를 사용하기 위한 조건으로 옳은 것은? → **정답: logits shape `(batch_size, num_classes)`, target shape `(batch_size,)`, target dtype `torch.long`**. one-hot 인코딩이 아니라 class 번호(index) Tensor를 그대로 사용
4. `random_split`으로 나눈 split을 실행할 때마다 재현하려면? → **정답: seed를 고정한 `torch.Generator().manual_seed(...)`를 `generator` 인자로 넘긴다**. generator 없이 호출하면 실행마다 split 결과가 달라짐
5. train loop와 validation loop 차이 설명 중 옳지 않은 것은? → **정답(틀린 설명): "validation loop에서는 `optimizer.step()`을 호출해 검증 데이터로도 학습을 진행한다"는 설명이 틀림**. validation loop에서 `optimizer.step()`을 호출하면 검증 데이터로 모델이 학습되어 검증의 의미(학습에 쓰지 않은 데이터로 평가) 자체가 사라지므로 사용하면 안 됨

**한 줄 정리**: transform은 `Resize→ToTensor→Normalize` 순서로 원본 데이터를 모델 입력에 맞게 바꾸되 랜덤 augmentation은 train에만 넣어야 하고, `random_split`은 원본 Dataset을 복사하지 않고 index만 나누므로 split별로 다른 transform을 쓰려면 `SubsetWithTransform` 같은 Wrapper로 index와 transform을 함께 관리해야 하며, 이렇게 구성한 train/valid/test 파이프라인 위에서 `nn.Module`(`__init__`+`forward`)로 만든 MLP를 `zero_grad→forward→loss→backward→step` 학습 루프와 `eval+no_grad` 검증 루프로 돌리고 best valid 시점의 `state_dict`를 복원해 test는 마지막에 딱 한 번만 확인하는 것이 오늘 완성한 전체 흐름이다.
