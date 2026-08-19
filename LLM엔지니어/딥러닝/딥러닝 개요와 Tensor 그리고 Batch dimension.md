# 2026-08-19 딥러닝 기초

## 오늘의 TIL 요약

**1장. 딥러닝 개요와 학습 파이프라인**
- 딥러닝 학습은 `데이터 → 모델 forward → loss 계산 → backward → optimizer.step() → validation` 흐름으로 요약됨
- 전통 ML은 사람이 특징을 설계하지만, 딥러닝은 원본 입력에서 특징 표현까지 함께 학습
- 학습 step 5줄 패턴 고정: `zero_grad() → model(x) → loss_fn() → backward() → step()` (zero_grad 누락 시 gradient가 누적되므로 주의)
- `model.train()`/`model.eval()`은 gradient 계산 스위치가 아니라 Dropout/BatchNorm 등의 모드 전환용, 실제 gradient 차단은 `torch.no_grad()`가 담당
- 문제 유형(회귀/이진분류/다중분류/다중레이블)에 따라 출력층 shape·loss·target dtype이 달라짐 (예: CrossEntropyLoss는 long 타입 class index, BCEWithLogitsLoss는 float 0/1)
- 데이터 유형(tabular/image/text/시계열)별 입력 shape 규칙과 기본 모델 후보(MLP/CNN/RNN·LSTM/Transformer) 정리
- 코드를 10개 블록(import~checkpoint)으로 나눠 읽는 습관, train loss는 줄고 valid loss가 늘면 과적합 신호로 체크

**2장. Tensor와 Batch Dimension**
- Tensor는 값보다 먼저 shape·dtype·device를 확인해야 함
- 차원 개념: 0차원(scalar)~3차원(배치 이미지 등) shape 읽는 법 정리
- `unsqueeze(0)`으로 배치 차원 추가, `squeeze(0)`으로 제거 (인자 없는 `squeeze()`는 모든 1차원을 지워버릴 수 있어 위험)
- Broadcasting은 shape를 오른쪽부터 비교, 차원이 같거나 1이거나 없으면 맞춰짐 — `(batch_size, 1)`과 `(batch_size,)`는 다르게 동작할 수 있으니 loss 계산 전 shape 항상 확인

**한 줄 정리**: 오늘은 학습 루프의 5단계 패턴과, 문제 유형·데이터 유형에 따라 shape/loss/dtype이 어떻게 갈라지는지, 그리고 Tensor의 batch dimension과 broadcasting 규칙을 학습함.

---

## 1장: 딥러닝 개요와 학습 파이프라인

### 1장1강: 딥러닝 기초 학습 로드맵

문제 정의
-> 데이터 준비
-> 입력과 정답 구성
-> 모델 설계
-> 손실 함수 선택
-> 옵티마이저 선택
-> 학습 loop 실행
-> 검증 loop 실행
-> 성능 기록
-> 모델 저장
-> 추론

앞으로의 핵심은 코드를 외우는 것이 아니라 데이터 -> 모델 -> 손실 -> 최적화 -> 평가 흐름을 읽는 힘을 기르는 것입니다.

### 1장2강: 머신러닝과 딥러닝의 차이

전통적인 머신러닝: 사람이 입력 특징을 주로 설계하고 모델이 그 특징의 패턴을 학습합니다.
딥러닝: 여러 층이 원본 입력에서 예측에 유용한 특징 표현까지 함께 학습합니다.

Private LLM 트랙에서는 이후 아래와 같은 문제로 연결될 수 있습니다.
- 고객 문의 intent classification
- 도메인 문서 분류
- RAG 검색 결과 relevance classifier
- 답변 안전성/개인정보 포함 여부 분류기
- 문서 요약 품질 평가 모델

실무에서는 먼저 간단한 baseline을 만들고, 딥러닝이 실제로 개선을 만드는지 비교해야 합니다.

### 1-2강: 데이터-모델-손실-최적화-평가 흐름

딥러닝 코드는 처음 보면 길고 복잡해 보입니다.
하지만 핵심 흐름은 아래와 같습니다.
데이터를 꺼냅니다.
-> 모델에 넣어 예측값을 만듭니다. (forward)
-> 예측값과 정답의 차이를 loss로 계산합니다.
-> loss를 기준으로 gradient를 계산합니다.
-> optimizer가 parameter를 업데이트합니다.
-> validation 데이터로 성능을 확인합니다.

**Dataset**
-> 샘플과 label을 저장합니다.

**DataLoader**
-> Dataset에서 데이터를 batch 단위로 꺼냅니다.

**loss 계산**
```python
loss = loss_fn(preds, batch_y)
```

| 문제 유형 | 대표 손실 함수 |
| --- | --- |
| 회귀 | nn.MSELoss() |
| 이진 분류 | nn.BCEWithLogitsLoss() |
| 다중 분류 | nn.CrossEntropyLoss() |

**backward와 optimizer step**

| 용어 | 쉽게 말하면 | 코드에서의 역할 |
| --- | --- | --- |
| 파라미터(parameter) | 모델 내부에 들어 있는 조절 가능한 숫자입니다. | weight, bias처럼 학습을 통해 값이 바뀝니다. |
| gradient | 각 파라미터를 조금 바꿀 때 loss가 어느 방향으로 얼마나 민감하게 변하는지를 나타내는 값입니다. | loss.backward()가 계산하고, optimizer가 보통 그 반대 방향으로 이동합니다. |
| optimizer | gradient를 보고 실제로 파라미터 값을 업데이트하는 도구입니다. | optimizer.step()이 업데이트를 수행합니다. |

학습 step의 핵심 순서는 다음과 같습니다.
1. optimizer.zero_grad()
2. preds = model(batch_x)
3. loss = loss_fn(preds, batch_y)
4. loss.backward()
5. optimizer.step()

| 코드 | 의미 |
| --- | --- |
| optimizer.zero_grad() | 이전 batch에서 계산된 gradient를 초기화합니다. |
| model(batch_x) | 현재 batch에 대한 예측값을 만듭니다. |
| loss_fn(preds, batch_y) | 예측값과 정답의 차이를 계산합니다. |
| loss.backward() | loss를 기준으로 gradient를 계산합니다. |
| optimizer.step() | gradient를 사용해 parameter를 업데이트합니다. |

학습 loop에서 optimizer.zero_grad()로 gradient를 재설정하고, loss.backward()로 손실을 역전파한 뒤, optimizer.step()으로 파라미터를 조정합니다.

이 그림은 cost function을 최소화하기 위해 gradient descent가 수렴 방향으로 이동하는 모습을 보여줍니다.

> ⛔ 주의사항
> optimizer.zero_grad()를 빼먹으면 gradient가 이전 batch의 값과 누적됩니다. 의도적인 gradient accumulation이 아니라면 매 batch마다 초기화해야 합니다.

**train loop와 validation loop**

train loop는 모델이 실제로 학습되는 구간입니다.
train 데이터 사용
-> loss 계산
-> gradient 계산
-> parameter 업데이트

이때는 보통 아래 코드를 사용합니다.
```python
model.train()
```

model.train()은 gradient 계산을 켜는 명령이 아니라, Dropout이나 BatchNorm 같은 층을 학습 모드로 바꾸는 명령입니다. 실제 gradient 계산 여부는 autograd와 torch.no_grad()가 결정합니다.

validation loop는 모델 parameter를 업데이트하지 않습니다.
학습 데이터의 gradient 업데이트에 직접 쓰지 않은 validation 데이터로 성능을 확인하고, 모델 선택이나 하이퍼파라미터 조정에 활용합니다. 반복해서 참고할 수 있는 validation 데이터는 최종 성능을 한 번 확인하는 test 데이터와 역할이 다릅니다.

validation 데이터 사용
-> 예측
-> loss/metric 계산
-> parameter 업데이트 없음

이때는 보통 아래 코드를 사용합니다.
```python
model.eval()

with torch.no_grad():
    ...
```

train loop에서는 model.train()을 사용하고, validation loop에서는 model.eval()로 Dropout·BatchNorm 등을 평가 모드로 바꾼 뒤 torch.no_grad()로 gradient 기록을 꺼서 메모리 사용을 줄입니다. 두 기능은 역할이 서로 다릅니다.

**연습 문제: 학습 파이프라인 순서 카드 정렬**
1. DataLoader에서 batch 꺼내기
2. optimizer.zero_grad()
3. model로 예측값 생성
4. loss_fn으로 loss 계산
5. loss.backward()
6. optimizer.step()
7. validation loss 계산

**오늘의 정리**
이번 시간에는 딥러닝 학습 파이프라인의 핵심 흐름을 배웠습니다.
- DataLoader는 batch 단위로 데이터를 제공합니다.
- model forward는 입력을 예측값으로 바꾸는 과정입니다.
- loss는 예측값과 정답의 차이를 숫자로 표현합니다.
- loss.backward()는 gradient를 계산합니다.
- optimizer.step()은 parameter를 업데이트합니다.
- validation loop는 성능 확인을 위해 사용합니다.

### 1-3강: 딥러닝 문제 유형과 입출력 구조 설계

딥러닝에서 모델 구조는 문제 유형과 분리해서 생각할 수 없습니다.
문제 유형을 잘못 이해하면 출력층 shape, loss, metric이 모두 어긋납니다.

| 질문 | 문제 유형 |
| --- | --- |
| 다음 달 구매 금액은 얼마인가요? | 회귀 |
| 이 고객이 이탈할까요? | 이진 분류 |
| 이 문의는 배송/환불/결제/계정 중 어디에 속하나요? | 다중 분류 |

문제 유형이 바뀌면 모델 마지막 layer도 달라집니다. logit은 Sigmoid나 Softmax를 적용하기 전, 모델이 만든 원시 점수입니다.
- 회귀: 필요한 개수의 실수값 출력
- 이진 분류: logit 1개 출력
- 다중 클래스 분류: class 개수만큼 logits 출력

손실 함수와 정답 Tensor 형식도 달라집니다.
- 회귀: float target + MSELoss
- 이진 분류: 0/1 float target + BCEWithLogitsLoss
- 다중 클래스 분류: class index long target + CrossEntropyLoss

다중 클래스와 다중 레이블을 구분하세요.
- 다중 클래스: 여러 class 중 하나를 고릅니다. 출력은 class별 logits, loss는 보통 CrossEntropyLoss입니다.
- 다중 레이블: 한 샘플에 여러 label이 동시에 참일 수 있습니다. label마다 logit 1개를 출력하고 보통 BCEWithLogitsLoss를 사용합니다.

> ⛔ 주의사항
> BCEWithLogitsLoss를 사용할 때는 모델 마지막에 Sigmoid를 직접 넣지 않는 것이 일반적입니다. 이 손실 함수가 Sigmoid와 BCE를 함께 처리하기 때문입니다.

**데이터 유형별 입력 구조**

딥러닝에서는 데이터 유형에 따라 입력 Tensor의 shape이 달라집니다.

| 데이터 유형 | 예시 | 일반적인 입력 shape |
| --- | --- | --- |
| Tabular | 고객 정보, 거래 데이터 | [batch_size, feature_dim] |
| Image | 상품 이미지, 의료 이미지 | [batch_size, channels, height, width] |
| 텍스트 token ID | 정수로 바꾼 문장 | [batch_size, sequence_length] (torch.long) |
| 텍스트 embedding | token별 실수 벡터 | [batch_size, sequence_length, embedding_dim] |
| 시계열 | 시간별 센서·수치 feature | [batch_size, sequence_length, feature_dim] |

**모델 선택 흐름**

처음에는 아래 기준으로 모델 후보를 생각하면 됩니다.

| 데이터 | 기본 모델 후보 | 이유 |
| --- | --- | --- |
| 표 데이터 | MLP | feature vector를 입력으로 사용하기 쉽습니다. |
| 이미지 데이터 | CNN | 공간적 패턴과 지역 특징을 활용합니다. |
| 시퀀스 데이터 | RNN/LSTM | 순서 정보를 hidden state로 전달합니다. |
| 긴 텍스트/문맥 | Transformer | 긴 문맥과 병렬 처리를 더 잘 다룹니다. |

**오늘의 정리**
이번 시간에는 문제 유형과 입출력 구조를 연결했습니다.
- 회귀는 연속적인 숫자를 예측합니다.
- 이진 분류는 두 class 중 하나를 예측합니다.
- 다중 분류는 여러 class 중 하나를 예측합니다.
- 문제 유형에 따라 출력층 shape, loss, metric이 달라집니다.
- 데이터 유형에 따라 MLP, CNN, RNN/LSTM 같은 모델 후보가 달라집니다.

### 1-4강: 기본 코드 구조 읽기

먼저 해야 할 일은 코드의 큰 덩어리를 나누는 것입니다.
1. import
2. config / hyperparameter
3. dataset / dataloader
4. model
5. loss function
6. optimizer
7. train loop
8. validation loop
9. metric
10. checkpoint / logging

코드를 읽을 때는 아래 질문을 순서대로 던지면 좋습니다.
- 이 코드는 어떤 라이브러리를 쓰나요?
- 데이터는 어디서 오나요?
- 입력과 정답 shape은 무엇인가요?
- 모델은 어떤 구조인가요?
- loss는 무엇인가요?
- optimizer는 무엇인가요?
- train loop에서 parameter가 업데이트되나요?
- validation loop에서 gradient 계산을 끄고 있나요?
- metric은 어디서 계산하나요?
- 모델은 어디에 저장하나요?

**데이터 영역 읽기**
```python
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

확인해야 할 질문은 다음과 같습니다.
- X는 어떤 shape인가요?
- y는 어떤 shape인가요?
- batch_size는 얼마인가요?
- shuffle은 True인가요?
- train과 validation이 나누어져 있나요?

DataLoader는 Dataset에서 한 번에 하나씩 가져오는 feature와 label을 minibatch로 전달하고, epoch마다 데이터를 섞는 등의 복잡성을 추상화해 줍니다.

**모델 영역 읽기**
```python
model = nn.Sequential(
    nn.Linear(4, 16), # 입력층(input)
    nn.ReLU(), # 은닉층(hidden)
    nn.Linear(16, 1), # 출력층(output)
)
```

nn.Sequential은 순서가 있는 module container이며, 데이터가 정의된 순서대로 module을 통과합니다.

**loss와 optimizer 영역 읽기**

loss와 optimizer는 학습 목표와 업데이트 방식을 결정합니다.

```python
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
```

| 코드 | 의미 |
| --- | --- |
| nn.MSELoss() | 회귀 문제에서 예측값과 정답의 평균제곱오차를 계산합니다. |
| model.parameters() | optimizer가 업데이트할 parameter 목록입니다. |
| lr=0.05 | learning rate입니다. 한 번 업데이트할 때 이동하는 크기를 의미합니다. |

**train/valid loop 영역 읽기**

train loop에서 반드시 확인할 코드는 다음입니다.
```python
model.train()
optimizer.zero_grad()
preds = model(batch_x)
loss = loss_fn(preds, batch_y)
loss.backward()
optimizer.step()
```

validation loop에서는 다음을 확인합니다.
```python
model.eval()

with torch.no_grad():
    preds = model(batch_x)
```

model.eval()은 Dropout·BatchNorm 같은 층을 평가 모드로 바꾸고, torch.no_grad()는 gradient 기록을 끕니다. validation loop에서는 parameter를 업데이트하지 않습니다.

**metric과 checkpoint 영역 읽기**

metric은 보통 epoch마다 기록합니다.
```python
history.append({
    "epoch": epoch,
    "train_loss": train_loss,
    "valid_loss": valid_loss,
})
```

학습이 잘 되고 있는지 확인하려면 loss와 metric을 함께 봐야 합니다.
- train loss는 줄어드나요?
- valid loss도 줄어드나요?
- train loss만 줄고 valid loss는 증가하지 않나요?

checkpoint는 모델을 나중에 다시 불러오기 위해 저장하는 파일입니다.
```python
torch.save(model.state_dict(), "model.pt")
```

- model_state_dict
- optimizer_state_dict
- epoch
- best_valid_loss
- config

**코드스니펫: 구조 해석용 학습 코드**
```python
# 1. import
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# 2. config / hyperparameter
SEED = 42
BATCH_SIZE = 32
LR = 0.05
EPOCHS = 5

torch.manual_seed(SEED)


# 3. data
X = torch.randn(200, 4)
true_w = torch.tensor([[2.0], [-1.0], [0.5], [3.0]])
y = X @ true_w + 0.1 * torch.randn(200, 1)

dataset = TensorDataset(X, y)
train_dataset, valid_dataset = random_split(dataset, [160, 40])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 4. device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 5. model
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
).to(device)


# 6. loss / optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


# 7. train function
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


# 8. validation function
@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


# 9. epoch loop / logging
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    valid_loss = validate(model, valid_loader, loss_fn, device)

    log = {
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }
    history.append(log)

    print(
        f"Epoch {epoch} | "
        f"train_loss={train_loss:.4f} | "
        f"valid_loss={valid_loss:.4f}"
    )


# 10. checkpoint
torch.save(model.state_dict(), "basic_mlp_regression.pt")
```

**연습문제: 빈칸에 단계명 채우기**
1. ____________________: 필요한 라이브러리를 불러옵니다.
2. ____________________: batch size, learning rate, epoch 수를 정의합니다.
3. ____________________: X, y를 만들고 DataLoader를 구성합니다.
4. ____________________: 입력을 예측값으로 바꾸는 신경망을 정의합니다.
5. ____________________: 예측값과 정답의 차이를 계산합니다.
6. ____________________: parameter를 업데이트하는 방법을 정합니다.
7. ____________________: 학습 데이터로 parameter를 업데이트합니다.
8. ____________________: 검증 데이터로 성능을 확인합니다.
9. ____________________: epoch별 loss와 metric을 저장합니다.
10. ____________________: 학습된 모델을 파일로 저장합니다.

1. import
2. config / hyperparameter
3. data / dataloader
4. model
5. loss function
6. optimizer
7. train loop
8. validation loop
9. logging
10. checkpoint

## 2장: Tensor와 Batch Dimension

### 2장1강: Tensor 생성과 dtype/shape 확인

딥러닝 모델은 숫자를 입력받고 숫자를 출력합니다.
이미지도 숫자, 텍스트도 숫자, 정답 라벨도 숫자로 바꿔서 모델에 넣습니다.
이때 PyTorch에서 숫자 데이터를 담는 기본 그릇이 Tensor입니다.
딥러닝에서 Tensor를 볼 때는 값 자체보다 먼저 shape, dtype, device를 확인해야 합니다.

**Tensor의 차원 이해하기**

Tensor의 차원은 "몇 겹으로 숫자가 감싸져 있는가"를 뜻합니다.
- 0차원 Tensor: Scalar (숫자값 하나) → `tensor(7)`
- 1차원 Tensor: Vector (1차원 배열) → `tensor([1, 2, 3])`
- 2차원 Tensor: Matrix (행과 열이 있는 표 형태) → `tensor([[1, 2, 3], [4, 5, 6]])`
- 3차원 Tensor: 여러 개의 행렬 묶음 (2차원 표가 여러 장 쌓인 구조)
  ```python
  torch.tensor([
      [[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]
  ])
  # torch.Size([2, 2, 3])
  ```

**shape을 읽는 기본 규칙**

| shape | 읽는 방법 | 예시 |
| --- | --- | --- |
| torch.Size([]) | 값 하나 | scalar |
| torch.Size([3]) | 값 3개 | vector |
| torch.Size([2, 3]) | 2행 3열 | matrix |
| torch.Size([10, 4]) | 샘플 10개, 특성 4개 | tabular batch |
| torch.Size([32, 3, 224, 224]) | 이미지 32장, 채널 3개, 높이 224, 너비 224 | image batch |

torch.Size([32, 3, 224, 224])라면 가장 바깥에 32개가 있고, 그 안에 3개가 있고, 그 안에 224개, 그 안에 224개가 있는 구조입니다.

**Tensor 속성 확인하기**

PyTorch Tensor는 값만 가지고 있는 것이 아닙니다.
Tensor는 다음 정보를 함께 가지고 있습니다.

| 속성 | 의미 | 예시 |
| --- | --- | --- |
| shape | Tensor의 모양 | torch.Size([3, 4]) |
| ndim | Tensor의 차원 수 | 2 |
| dtype | 값의 자료형 | torch.float32 |
| device | Tensor가 저장된 장치 | cpu, cuda:0 |

딥러닝에서 자주 보는 dtype은 다음과 같습니다.

| dtype | 주로 쓰는 곳 |
| --- | --- |
| torch.float32 | 입력 데이터, 모델 가중치, 회귀 정답 |
| torch.float64 | 더 정밀한 실수 연산이 필요한 경우 |
| torch.int64 또는 torch.long | CrossEntropyLoss에서 클래스 인덱스로 쓰는 라벨 |
| torch.bool | 조건 마스크 |

예를 들어 이미지 픽셀 값, 임베딩 벡터, 모델의 가중치는 보통 실수입니다.
반면 CrossEntropyLoss로 다중 클래스 분류를 할 때의 정답은 보통 정수 클래스 번호입니다. 이진 분류의 BCEWithLogitsLoss target은 예외로, 0/1 값을 가진 실수형 Tensor를 사용합니다.

> ⛔ 주의사항
> 분류 target의 dtype은 loss 함수에 따라 다릅니다.
> - CrossEntropyLoss + class index target: torch.long
> - BCEWithLogitsLoss + 0/1 target: torch.float32 등 입력과 같은 실수형
> 같은 '분류 라벨'이라도 loss가 기대하는 형식을 먼저 확인해야 합니다.

### 2장2강: Batch dimension과 Broadcasting

딥러닝에서 2차원 입력 Tensor는 보통 (batch_size, features) 형태입니다.

**이미지 데이터의 batch dimension**

이미지 데이터는 보통 다음 형태를 사용합니다.
`(batch_size, channels, height, width)`

**unsqueeze로 batch 차원 추가하기**

샘플 하나만 있을 때도 모델은 batch 형태를 기대할 수 있습니다.
이럴 때는 unsqueeze로 차원을 하나 추가합니다.
```python
# 0번째 위치에 차원을 하나 추가합니다.
# (4,) -> (1, 4)
sample_batch = sample.unsqueeze(0)
```

**squeeze로 크기가 1인 차원 제거하기**

squeeze는 크기가 1인 차원을 제거합니다.
```python
# 크기가 1인 차원을 제거합니다.
x = torch.randn(1, 4)
# (1, 4) -> (4,)
y = x.squeeze(0)
```

> ⛔ 주의사항
> squeeze()를 인자 없이 사용하면 크기가 1인 모든 차원을 제거합니다.
> 예를 들어 (1, 1, 4)에서 모든 1 차원이 사라져 (4,)가 될 수 있습니다. 실습에서는 가능하면 squeeze(0)처럼 제거할 차원을 명시하세요.

**Broadcasting이란?**

Broadcasting은 서로 다른 shape의 Tensor를 연산할 때, PyTorch가 가능한 경우 자동으로 크기를 맞춰주는 규칙입니다.

두 Tensor가 broadcastable 하려면 뒤쪽 차원부터 비교했을 때 각 차원이 같거나, 둘 중 하나가 1이거나, 한쪽 차원이 존재하지 않아야 합니다. 결과 shape는 각 차원에서 더 큰 값을 사용합니다.

Broadcasting은 shape를 오른쪽부터 비교합니다.
예를 들어 다음 두 Tensor를 봅니다.
```
x shape:    (2, 3, 4)
y shape:       (3, 1)
```

차원 수가 다르면 짧은 쪽 앞에 1이 있다고 생각합니다.
```
x shape:    (2, 3, 4)
y shape:    (1, 3, 1)
```

이제 오른쪽부터 비교합니다.
결과 shape는 각 위치의 큰 값입니다.
```
result shape: (2, 3, 4)
```

> ⛔ 주의사항
> (batch_size, 1)과 (batch_size,)는 비슷해 보이지만 broadcasting에서는 완전히 다르게 동작할 수 있습니다.
> Loss를 계산하기 전에는 항상 예측값과 정답값의 shape를 출력하세요.
