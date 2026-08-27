# 2026-08-27 실험 재현성과 checkpoint 관리 그리고 과적합·과소적합 진단

## 오늘의 TIL 요약

**9장1강. seed 고정과 재현성 요소**
- 재현성(reproducibility): 같은 조건에서 실험을 다시 실행했을 때 같거나 비슷한 결과를 얻을 수 있는 성질 — 실험 비교, 디버깅, 리포트 작성, 팀 협업의 전제 조건
  - 재현성이 없으면 "hidden_dim을 키워서 좋아진 것"인지 "이번 실행이 운이 좋았던 것"인지 구분할 수 없음
- 딥러닝 코드에서 randomness가 끼어드는 대표적인 위치
  | 위치 | 예시 |
  | --- | --- |
  | Python `random` | Python 기본 난수 |
  | NumPy `random` | NumPy 기반 샘플링 |
  | PyTorch random | Tensor 생성, weight 초기화 |
  | CUDA random | GPU 연산 및 CUDA 난수 |
  | DataLoader shuffle | batch 순서 섞기 |
  | `random_split` | train/valid 데이터 분할 |
  | Dropout, 랜덤 augmentation | 학습 중 매 forward마다 새로 뽑는 난수 |
- 그래서 seed를 **하나만** 고정하는 것으로는 부족하고, 여러 난수 생성기를 함께 고정해야 함
  ```python
  import random
  import numpy as np
  import torch

  def set_seed(seed=42):
      """Python, NumPy, PyTorch의 random seed를 고정합니다."""
      random.seed(seed)                    # Python 기본 random 모듈
      np.random.seed(seed)                 # NumPy
      torch.manual_seed(seed)              # PyTorch CPU
      if torch.cuda.is_available():        # GPU를 쓸 수 있다면 CUDA도
          torch.cuda.manual_seed(seed)     # 현재 GPU
          torch.cuda.manual_seed_all(seed) # 모든 GPU
  ```
- ⛔ seed를 고정해도 **모든 환경에서 완전히 같은 결과가 항상 보장되지는 않음**
  - PyTorch release/commit/platform이 다르면 보장되지 않고, CPU와 GPU 사이에서도 결과가 다를 수 있음
  - GPU 연산이 비결정적인 이유: 부동소수점 덧셈은 결합법칙이 성립하지 않는데(`(a+b)+c != a+(b+c)`), GPU는 수천 개 thread가 `atomicAdd` 등으로 **누적 순서가 실행마다 달라지는 방식**으로 합을 구함 → 마지막 자리 수준의 오차가 생기고, 그 오차가 epoch를 거치며 증폭될 수 있음
  - cuDNN이 실행할 때마다 더 빠른 알고리즘을 자동 탐색(autotuning)하는 것도 결과가 흔들리는 원인
- (심화) 결정성을 더 강하게 요구하고 싶을 때 — 대신 속도는 느려질 수 있음
  ```python
  torch.backends.cudnn.deterministic = True   # 결정적 cuDNN 알고리즘만 사용
  torch.backends.cudnn.benchmark = False      # 알고리즘 자동 탐색 끄기
  torch.use_deterministic_algorithms(True)    # 비결정적 연산을 만나면 에러로 알려줌
  # 일부 CUDA 연산은 환경변수도 필요: CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```
- split과 shuffle에는 seed를 고정한 `torch.Generator`를 넘겨 재현성을 확보
  ```python
  split_generator = torch.Generator().manual_seed(42)
  train_part, valid_part = random_split(dataset, [0.8, 0.2], generator=split_generator)

  loader_generator = torch.Generator().manual_seed(42)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                            generator=loader_generator)
  ```
- 이 교안은 **Colab/CPU + `num_workers=0`** 기준 — Dataset과 transform이 메인 프로세스에서 실행되므로 별도의 `worker_init_fn`이 필요 없음
  - `num_workers>0`으로 바꾸면 worker가 각자 다른 프로세스에서 돌기 때문에 worker별 난수 초기화(`worker_init_fn`)와 라이브러리별 난수 사용을 따로 점검해야 함
- ⭐ **"정확한 재개"의 경계**
  - 이 과정에서 "정확히 재개"는 **한 epoch의 모든 batch 처리와 history 기록이 끝난 뒤 저장한 checkpoint에서 다음 epoch를 시작하는 경우만** 의미
  - 같은 코드·데이터·분할·라이브러리·장치 환경을 쓰고, **다음 DataLoader iterator를 만들기 전에** 상태를 복원해야 함
  - epoch **도중** 중단된 지점의 다음 batch부터 이어 가는 기능은 기초 과정 범위 밖 → 미완료 epoch는 처음부터 다시 실행
- ⛔ 자주 하는 실수: **재개할 때 seed 값만 다시 지정하기**
  - `set_seed(42)`를 다시 부르면 난수열의 **처음**으로 돌아갈 뿐, checkpoint 저장 시점의 난수 상태로 돌아가지 않음
  - 완료된 epoch 다음부터 정확히 재개하려면 model/optimizer뿐 아니라 **난수 상태와 DataLoader generator 상태**까지 저장·복원해야 하고, PyTorch 버전·장치·데이터 분할·주요 hyperparameter도 함께 기록해야 함

**9장2강. logging 설계**
- 8장에서는 metric을 `history` dictionary에 담았지만, notebook을 닫거나 런타임이 초기화되면 메모리에 있던 history는 사라짐 → 중요한 실험 결과는 **파일로** 남겨야 함
  | 로그 | 저장 파일 | 의미 |
  | --- | --- | --- |
  | config log | `config.json` | 어떤 설정으로 실험했는지 |
  | metric log | `metrics.csv` | epoch별 loss/accuracy |
- 왜 두 파일로 나누는가: config는 **한 번 쓰고 끝나는 설정 스냅샷**(중첩 구조를 표현하기 좋은 JSON), metric은 **epoch마다 한 줄씩 늘어나는 표**(pandas/Excel로 바로 읽히는 CSV)
- 무엇을 기록해야 하나
  | 범주 | 예시 |
  | --- | --- |
  | 데이터 | 데이터 출처·버전, split seed·크기 또는 인덱스, train/valid/test transform |
  | 모델 | `hidden_dim`, `num_layers`, dropout 사용 여부 |
  | 학습 설정 | `batch_size`, `learning_rate`, optimizer, `epochs` |
  | 결과 | `train_loss`, `valid_loss`, `train_acc`, `valid_acc` |
  | 저장 정보 | best checkpoint 경로 |
  | 환경 | Python·PyTorch 버전, device, seed |
- config 저장: 한글이 깨지지 않게 `ensure_ascii=False`, 사람이 읽기 쉽게 `indent=2`
  ```python
  import json

  config = {
      "seed": 42, "hidden_dim": 128, "batch_size": 32,
      "learning_rate": 1e-3, "optimizer": "Adam", "epochs": 5,
      "device": str(device), "torch_version": torch.__version__,
  }
  with open("config.json", "w", encoding="utf-8") as f:
      json.dump(config, f, ensure_ascii=False, indent=2)
  ```
- metric 저장: epoch마다 **한 줄씩 추가**해야 하므로 `mode="a"`(append). header는 파일이 없을 때 한 번만 씀
  ```python
  import csv, os

  def log_metrics(path, row):
      write_header = not os.path.exists(path)
      # newline="" 를 빼면 Windows에서 빈 줄이 한 줄씩 끼어듦
      with open(path, "a", newline="", encoding="utf-8") as f:
          writer = csv.DictWriter(f, fieldnames=list(row.keys()))
          if write_header:
              writer.writeheader()
          writer.writerow(row)

  log_metrics("metrics.csv", {
      "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
      "valid_loss": valid_loss, "valid_acc": valid_acc,
  })
  ```
  - `"w"`는 파일을 새로 쓰므로 이전 epoch 기록이 사라지고, `"r"`은 읽기 전용, `"x"`는 파일이 이미 있으면 에러 → **추가 기록은 `"a"`**
- 화면 출력(`print`)은 학습 중 확인용, 파일 저장은 사후 분석용 → 둘을 함께 쓰는 것이 기본
- (심화) 규모가 커지면 `print` 대신 Python `logging` 모듈(레벨/포맷/파일 핸들러), 더 나아가 TensorBoard·Weights & Biases 같은 실험 추적 도구를 쓰지만, **"설정과 결과를 실행 밖에 남긴다"**는 아이디어는 동일

**9장3강. state_dict와 checkpoint 구성**
- 학습한 모델을 저장할 때는 **모델 객체 전체**가 아니라 학습된 파라미터를 담은 **`state_dict`**를 저장하는 방식을 주로 사용
  - `state_dict`는 `{"fc1.weight": Tensor, "fc1.bias": Tensor, ...}` 형태의 dictionary
  - 모델 객체 전체(`torch.save(model)`)는 pickle이 **클래스 정의 위치(모듈 경로)**를 함께 저장하므로, 나중에 파일을 옮기거나 클래스 이름을 바꾸면 로드가 깨짐 → `state_dict` 저장이 훨씬 유연
  | 객체 | `state_dict`에 들어가는 것 |
  | --- | --- |
  | model | weight, bias, BatchNorm running mean/var 등 |
  | optimizer | learning rate, momentum, Adam 내부 상태 등 |
- 목적별로 저장해야 할 정보가 다름
  | 목적 | 필요한 정보 |
  | --- | --- |
  | 추론만 하기 | `model_state_dict` + 모델 구조를 다시 만들 config |
  | 일반적인 학습 재개 | model, optimizer, `completed_epochs`, history, best metric, 핵심 config |
  | 완료 epoch 경계에서 정확히 재개 | 일반 재개 항목 + Python/NumPy/PyTorch CPU·CUDA 난수 상태 + DataLoader generator 상태 + EarlyStopping 상태 |
- `completed_epochs` = **모든 batch 처리와 history 기록까지 끝난 epoch 수**
  - 5 epoch를 모두 마치고 저장했다면 `completed_epochs=5`, 로드 후에는 **6번째 epoch부터** 시작
- ⭐ **저장 시점 계약**: 정확한 재개용 checkpoint는 반드시 epoch 전체가 끝나고 history를 갱신한 **뒤에** 저장. epoch 도중 중단됐다면 그 미완료 epoch는 처음부터 다시 실행
  ```python
  checkpoint = {
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "completed_epochs": epoch,          # 방금 끝낸 epoch 번호
      "history": history,
      "best_valid_loss": best_valid_loss,
      "config": config,
      # --- 정확한 재개용 상태 ---
      "python_rng": random.getstate(),
      "numpy_rng": np.random.get_state(),
      "torch_rng": torch.get_rng_state(),
      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
      "loader_generator": loader_generator.get_state(),
  }
  torch.save(checkpoint, "checkpoints/last.pt")
  ```
- ⛔ 자주 하는 실수: **optimizer 상태를 저장하지 않기**
  - Adam은 gradient의 1차/2차 moment(이동평균) 같은 내부 상태를 들고 학습하므로, 그 상태 없이 재개하면 **처음부터 다시 warm-up 하는 것과 비슷해져** 이어서 학습한 것과 결과가 달라짐
- (참고) PyTorch 2.6부터 `torch.load`의 `weights_only` 기본값이 `True` → RNG 상태처럼 순수 Tensor가 아닌 객체가 섞이면 로드가 막힐 수 있음. **본인이 만든 신뢰할 수 있는 파일**에 한해 `weights_only=False`를 명시하거나 `torch.serialization.add_safe_globals(...)`로 허용

**9장4강. 저장/불러오기와 학습 재개**
- 핵심 흐름: **checkpoint 불러오기 → 모델·optimizer 복원 → 다음 epoch 계산 → 학습 재개**
  | 목적 | 필요한 상태 |
  | --- | --- |
  | 추론 또는 평가 | `model_state_dict` |
  | 학습 재개 | `model_state_dict`, `optimizer_state_dict`, `completed_epochs` |
  ```python
  ckpt = torch.load("checkpoints/last.pt", map_location=device, weights_only=False)

  # 1) checkpoint의 config와 "같은 구조"로 모델을 먼저 만든 뒤 파라미터를 주입
  model = MLP(**ckpt["config"]["model"]).to(device)
  model.load_state_dict(ckpt["model_state_dict"])

  # 2) optimizer도 동일하게 생성한 뒤 상태 복원
  optimizer = torch.optim.Adam(model.parameters(), lr=ckpt["config"]["learning_rate"])
  optimizer.load_state_dict(ckpt["optimizer_state_dict"])

  # 3) 난수·generator 상태 복원 (iterator를 만들기 전에!)
  random.setstate(ckpt["python_rng"])
  np.random.set_state(ckpt["numpy_rng"])
  torch.set_rng_state(ckpt["torch_rng"])
  loader_generator.set_state(ckpt["loader_generator"])

  # 4) 다음 epoch부터 재개
  start_epoch = ckpt["completed_epochs"] + 1
  history = ckpt["history"]
  ```
  - `map_location=device`: GPU에서 저장한 checkpoint를 CPU에서 열 때처럼 **장치가 바뀌어도** 로드되게 해줌
- 자주 발생하는 실수
  | 실수 | 결과 | 해결 방법 |
  | --- | --- | --- |
  | 다른 모델 구조를 생성함 | parameter key 또는 shape 오류 | checkpoint config와 같은 구조로 생성 |
  | optimizer 상태를 불러오지 않음 | Adam 내부 상태가 사라짐 | 학습 재개 시 optimizer도 복원 |
  | history key나 길이가 다름 | `KeyError` 또는 기록 불일치 | metric key와 epoch 길이를 저장 전에 검사 |
  | 새 generator를 seed만 다시 지정함 | 중단 시점의 다음 shuffle과 달라짐 | 같은 generator 상태를 **iterator 생성 전에** 복원 |
  | `start_epoch = completed_epochs` | 마지막 epoch를 다시 실행함 | `completed_epochs + 1` 사용 |
  | 평가 시 `model.eval()`만 호출함 | gradient 추적은 계속 활성화됨 | `inference_mode()` 또는 `no_grad()`도 함께 사용 |
  | `load_state_dict("ckpt.pt")` | 경로 문자열을 그대로 넘김 | `torch.load()`로 dict를 먼저 읽고 그 안의 `model_state_dict`를 넘김 |
- `model.eval()`과 `no_grad()/inference_mode()`는 **역할이 다름** — `eval()`은 Dropout/BatchNorm의 **동작 모드**를 바꾸고, `no_grad()/inference_mode()`는 **gradient 계산과 메모리 사용**을 끔 → 평가에는 둘 다 필요
- 직접 확인해보기: ① 2 epoch checkpoint가 만들어지는지 확인 → ② 첫 재개 epoch가 **3**인지 확인 → ③ 총 5 epoch까지 학습한 뒤 checkpoint가 갱신되는지 확인 → ④ 최종 `completed_epochs`와 네 history 길이가 모두 **5**인지 확인
- 재개한 뒤에도 **매 epoch 끝에서 checkpoint를 다시 저장**해야 다음 중단에 대비할 수 있음

**9장5강. 실험 폴더 구조와 결과 관리**
- config·metric·checkpoint·plot이 여기저기 흩어지면 "이 그래프가 어떤 설정의 결과인지" 추적이 불가능해짐 → **한 번의 실험 = 하나의 폴더**
  ```
  runs/
  └── 20260826_153012_123456_mlp_baseline/
      ├── config.json
      ├── metrics.csv
      ├── checkpoints/
      │   ├── last.pt
      │   └── best.pt
      └── plots/
          └── loss_curve.png
  ```
  | 파일 또는 폴더 | 역할 |
  | --- | --- |
  | `config.json` | 모델 구조, learning rate, batch size, seed 등 실험 설정 |
  | `metrics.csv` | epoch별 train/validation loss와 accuracy |
  | `checkpoints/last.pt` | 마지막으로 **완료된** epoch의 학습 상태 |
  | `checkpoints/best.pt` | validation 성능이 가장 좋았던 모델 상태 |
  | `plots/loss_curve.png` | 학습 곡선을 이미지로 저장한 파일 |
- **`last.pt`는 학습 재개용, `best.pt`는 평가·추론용** — 마지막 epoch가 최고 성능인 경우는 오히려 드물기 때문에 둘을 반드시 구분해서 저장
- 폴더 이름에 **timestamp + 실험 이름**을 넣으면 실행할 때마다 새 폴더가 생겨 이전 실험을 덮어쓰지 않음
  ```python
  from datetime import datetime
  from pathlib import Path

  def create_run_dir(name, root="runs"):
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 초 단위 중복까지 방지
      run_dir = Path(root) / f"{stamp}_{name}"
      (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
      (run_dir / "plots").mkdir(parents=True, exist_ok=True)
      return run_dir

  run_dir = create_run_dir("mlp_baseline")
  ```
- 직접 확인해보기: `config.json`/`metrics.csv`/`last.pt`/`best.pt`/`loss_curve.png`가 모두 생성되는지 확인 → `metrics.csv`에 실제 3개 epoch 행이 저장됐는지 확인 → `last.pt`의 `completed_epochs`가 3인지 확인 → `create_run_dir("resume_test")`로 실험 이름을 바꾸고 `hidden_dim=64`로 수정해 **두 번째 run** 생성 → 두 run directory의 `config.json`과 `metrics.csv` 비교
- ⭐ 목표는 관리 함수를 길게 만드는 것이 아니라, **한 번의 실제 실험에서 나온 설정·metric·checkpoint·plot이 하나의 폴더로 연결되는 구조**를 확인하는 것
- (선택) 실무에서 추가로 남기는 정보: 데이터셋 이름·버전, train/valid/test split 정보, Python·PyTorch 버전, **Git commit ID**, 최종 결과를 요약한 `summary.json`, 여러 seed로 반복한 실험 결과, RNG·DataLoader generator 상태

**9장 전체 정리**
- seed는 실험의 random 조건을 관리하고, logging은 실제 설정과 epoch별 결과를 기록
- `last.pt`는 마지막 완료 epoch의 상태이고, `best.pt`는 validation 기준으로 선택한 상태
- resume은 저장한 **다음** epoch부터 학습을 이어감(`completed_epochs + 1`)
- run directory는 한 번의 실험에서 생성된 config·metrics·checkpoint·plot을 하나로 묶음
- 핵심은 **같은 코드를 실행하는 것에서 끝나지 않고, 실험 조건과 결과를 나중에 다시 확인할 수 있도록 남기는 것**

**10장1강. 학습 곡선 해석**
- 8장에서 만든 epoch 로그(history)를 **그래프로 바꿔서** 모델이 제대로 학습되고 있는지 판단하는 단계
  - loss curve와 accuracy curve를 **함께** 봐야 상태를 정확히 읽을 수 있음(loss는 예측의 확신 정도까지 반영하고 accuracy는 맞았는지 여부만 반영 → accuracy는 그대로인데 loss만 오르는 경우도 있음)
  ```python
  plt.plot(history["train_loss"], label="train")
  plt.plot(history["valid_loss"], label="valid")
  plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
  ```
- 곡선은 **방향(내려가고 있는가)**과 **간격(train과 valid의 gap)**을 함께 읽음. train과 valid의 차이를 일반화 gap(generalization gap)이라고 부름
- 위험 신호를 읽는 방법
  | 곡선 패턴 | 의심할 수 있는 문제 | 먼저 확인할 것 |
  | --- | --- | --- |
  | train loss만 계속 감소하고 valid loss는 증가 | 과적합 | Dropout, weight decay, Early Stopping |
  | train/valid loss가 모두 높고 거의 줄지 않음 | 과소적합 | 모델 크기, learning rate, epoch 수 |
  | loss가 크게 출렁임 | 불안정 학습 | learning rate, batch size |
  | validation accuracy가 이상하게 높음 | 데이터 누수 가능성 | train/valid split, 중복 샘플 |
  | loss가 `nan`으로 발산 | learning rate 과다, 수치 불안정 | lr 낮추기, gradient clipping, 입력 정규화 |
- ⛔ 실수 3. **validation 단계에서 `model.eval()`을 호출하지 않음** — Dropout과 BatchNorm은 `train()`과 `eval()`에서 동작이 다르므로 validation에서는 반드시 `model.eval()`을 호출해야 함
- ⛔ 실수 4. **학습 중 누적한 train metric과 valid metric을 그대로 같은 조건이라고 생각함**
  - train metric은 ① epoch 내내 **계속 바뀌는 weight**로 계산된 평균이고 ② Dropout이 켜져 있으며 ③ 랜덤 augmentation이 적용된 상태 → valid metric과 같은 잣대가 아님
  - 정밀한 일반화 gap이 필요하면 **epoch 마지막 weight를 eval mode로 고정**하고, 랜덤 증강이 없는 **평가용 train loader**로 train metric을 다시 계산

**10장2강. Overfitting과 Underfitting 진단**
- **Underfitting**: train 데이터조차 충분히 배우지 못한 상태 (train loss도 valid loss도 높음)
  | 원인 | 설명 | 대응 |
  | --- | --- | --- |
  | 모델이 너무 단순함 | hidden size가 작거나 layer 수가 부족 | 모델 용량 키우기 |
  | 학습 epoch가 너무 적음 | 충분히 배울 시간이 부족 | epoch 늘리기 |
  | learning rate가 너무 작음 | 파라미터가 너무 조금씩 바뀌어 학습이 느림 | lr 키우기 |
  | 입력 feature가 부족함 | 문제를 풀 정보 자체가 데이터에 없음 | feature 추가·데이터 재검토 |
- **Overfitting**: train에는 잘 맞지만 validation에는 일반화하지 못하는 상태 (train↓, valid↑ → gap이 벌어짐)
  | 원인 | 설명 | 대응 |
  | --- | --- | --- |
  | 모델이 너무 큼 | 데이터 수에 비해 parameter가 많음 | 모델 축소 |
  | epoch가 너무 많음 | 학습 데이터에 점점 더 맞춰짐 | Early Stopping |
  | regularization 부족 | Dropout, weight decay 등이 부족 | Dropout·weight decay 추가 |
  | 학습 데이터가 적거나 다양성이 부족함 | 훈련 샘플의 **우연한 패턴까지 외우기** 쉬워짐 | 데이터 추가, augmentation |
- 진단 순서: ① train loss가 충분히 낮아졌는가? (아니면 → Underfitting) → ② 낮아졌는데 valid loss가 오르는가? (그렇다면 → Overfitting) → ③ 둘 다 아닌데 성능이 이상하다면 split과 데이터 자체를 의심
- ⛔ **Overfitting과 distribution shift를 구분할 것**
  - train과 valid의 성능 차이가 크다고 해서 **항상** Overfitting인 것은 아님
  - train과 valid가 서로 다른 기간·장소·사람·장비에서 수집됐다면 두 데이터의 특성 자체가 다른 **distribution shift**가 원인일 수 있음
  - 예: train은 밝은 사진인데 valid는 어두운 사진 위주 → 작은 모델에서도 valid 성능이 크게 떨어짐. 이런 문제는 단순히 Dropout을 추가한다고 해결되지 않으므로 **먼저 데이터가 어떻게 나뉘었는지** 확인해야 함
- 대응 방법은 문제 유형에 맞게 **한 번에 하나씩** 적용하고 비교해야 무엇이 효과가 있었는지 알 수 있음 → 이 지점에서 9장의 run directory·config·metrics 기록이 그대로 필요해짐

**데일리 퀴즈 정리 (정답 포함)**
1. CSV 파일에 epoch metric을 한 줄씩 추가로 저장할 때 `open` 함수의 mode로 올바른 것은? → **정답: `"a"`**. `"w"`는 파일을 새로 쓰기 때문에 이전 내용이 사라지고, `"r"`은 읽기 전용이라 쓰기가 불가능하며, `"x"`는 파일이 이미 존재하면 오류를 발생시킴
2. 학습 재개(resume training)를 위한 checkpoint에 반드시 포함해야 할 정보는? → **정답: `model_state_dict`와 `optimizer_state_dict`, epoch 정보를 함께 저장**. 추론만 할 때는 `model_state_dict`만으로 충분하지만, 학습을 이어가려면 optimizer의 내부 상태(Adam의 moving average 등)와 다음 epoch를 계산할 epoch 정보가 필요
3. train loss는 계속 감소하지만 validation loss가 다시 증가하기 시작할 때 가장 의심할 수 있는 상태는? → **정답: Overfitting**. Underfitting은 train loss와 valid loss가 **모두** 높고 잘 줄지 않는 상태이므로 이 패턴과 다름
4. `torch.load`로 불러온 dictionary의 파라미터를 실제 모델 객체에 주입할 때 사용하는 메서드는? → **정답: `load_state_dict`**. `model.load_state_dict(ckpt["model_state_dict"])` 형태로 사용하며, `load_state_dict`에 **파일 경로 문자열을 직접 넣는 것**이 대표적인 실수 사례
5. Python·NumPy·PyTorch 등 여러 난수 생성기의 초기값을 동일하게 고정해 같은 코드를 다시 실행해도 비슷한 결과를 얻게 하는 성질은? → **정답: (실험) 재현성**. 실험 비교·디버깅·리포트 작성·팀 협업에서 중요하며, seed 고정은 재현성을 높이지만 **모든 환경에서 완전한 동일 결과를 항상 보장하지는 않음**

**한 줄 정리**: 재현 가능한 실험은 Python·NumPy·PyTorch·CUDA seed와 `random_split`/DataLoader의 `generator`까지 함께 고정하는 데서 시작하되 GPU 연산의 비결정성 때문에 완전한 동일 결과가 보장되지는 않으며, 설정은 `config.json`에 결과는 `metrics.csv`에 append(`"a"` mode)로 남기고, `model_state_dict`·`optimizer_state_dict`·`completed_epochs`·history·config(정확한 재개가 필요하면 RNG와 loader generator 상태까지)를 묶은 checkpoint를 매 epoch 끝에 저장해 `completed_epochs + 1`부터 재개하며, 이 산출물들을 timestamp 기반 run directory 하나로 묶어두면 학습 곡선에서 train loss만 내려가고 valid loss가 오르는 과적합 신호나 둘 다 높은 과소적합 신호를 발견했을 때 무엇을 바꿔서 무엇이 좋아졌는지 비교할 수 있게 된다.
