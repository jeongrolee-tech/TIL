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

**데일리퀴즈 정리**
1. `x.to(device)` 결과를 재할당하지 않으면 x는 그대로 CPU에 남음
2. `CrossEntropyLoss`의 class index target dtype은 `long`(int64)
3. `flatten(images, start_dim=1)`은 batch 차원 유지 → `(16,1,28,28)` → `(16,784)`
4. `nn.Linear(10,5)`의 weight shape는 `(5,10)` — (out_features, in_features) 순서, 내부 계산은 `input @ weight.T + bias`
5. MLP를 통과하며 batch size는 유지되고 feature 차원만 변함

**한 줄 정리**: device는 자동 이동이 안 되므로 `.to(device)` 결과를 재할당해 model/입력/라벨의 device를 통일해야 하고, shape 오류 디버깅에서는 batch 차원 유지가 핵심.
