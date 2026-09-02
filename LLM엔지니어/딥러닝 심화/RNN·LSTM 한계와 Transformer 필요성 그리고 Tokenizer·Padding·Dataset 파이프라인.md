# 2026-09-02 RNN·LSTM 한계와 Transformer 필요성 그리고 Tokenizer·Padding·Dataset 파이프라인

## 오늘의 TIL 요약

**1장1강. RNN/LSTM의 한계와 Transformer 필요성**
- 구조를 고르는 기준은 "Transformer가 더 최신이라서"가 아니라 **문맥을 전달하는 경로 길이와 계산 비용**
- **RNN의 계산 계약** — 이전 상태를 기다린다
  - `h_t = tanh(x_t·W_x + h_{t-1}·W_h + b)` → `x_1 → h_1 → h_2 → ... → h_L`
  - `h_t`가 `h_{t-1}`에 의존하므로 **같은 시퀀스의 위치 `t+1`은 위치 `t`의 계산이 끝나야** 진행 가능
  - ⛔ 흔한 오해: "RNN은 병렬 처리가 전혀 불가능하다"가 아님 — batch 안의 샘플과 한 시점의 행렬 연산은 병렬화됨. 남는 것은 **시간축 의존성 자체**
  ```python
  import torch

  # batch_first=True이므로 입력 축 순서는 [B, L, D_in]
  rnn = torch.nn.RNN(input_size=4, hidden_size=6, batch_first=True)
  x = torch.randn(2, 5, 4)      # [B=2, L=5, D_in=4]
  output, h_n = rnn(x)

  # output: (2, 5, 6)  모든 시점의 표현      [B, L, H]
  # h_n:    (1, 2, 6)  층·방향별 마지막 상태  [layers×directions, B, H]
  ```
  - ⛔ 다층·양방향에서는 `output[:, -1]`과 `h_n[-1]`을 **같은 뜻으로 단정하면 안 됨** (양방향이면 `output`의 마지막 위치는 정방향 마지막 + 역방향 첫 상태가 붙은 값)
- **긴 문맥에서 겹치는 세 문제** — 서로 다른 문제이므로 분리해서 설명해야 함

  | 문제 | 내용 | 남는 한계 |
  | --- | --- | --- |
  | 장기 의존성 | 첫 위치 → 마지막 위치 최소 경로가 대략 `L-1` | 문장이 길어질수록 단계가 함께 증가 |
  | Vanishing/Exploding gradient | 시간축 역전파에서 Jacobian이 반복 곱해짐 | Gradient clipping은 **폭주만 억제**, 장거리 관계를 학습시켜 주지는 않음 |
  | 고정 크기 상태의 정보 병목 | 초기 RNN Encoder–Decoder는 입력 전체를 마지막 상태 하나로 압축 | 입력이 길수록 세부 정보가 희석 |

  - 💡 Transformer 이전의 Attention도 원래는 **이 병목을 줄이려고** Encoder의 여러 상태를 직접 참고하도록 도입된 장치
- **LSTM이 바꾼 것과 바꾸지 못한 것**

  | 구성 | 역할 |
  | --- | --- |
  | Forget gate | 이전 정보 중 버릴 비율 결정 |
  | Input gate | 새 정보를 반영할 비율 결정 |
  | Output gate | 현재 출력에 드러낼 정보 결정 |
  | Cell state | 비교적 직접적인 정보 전달 경로 |

  - gradient 흐름은 개선하지만 **`t`가 `t-1`을 기다리는 순차성**과 **제한된 상태로 문맥을 축약하는 부담**은 그대로 남음
  - ⛔ "LSTM이 장기 의존성을 완전히 해결한다"고 표현하지 않음 — 시계열·스트리밍·작은 모델에서는 여전히 좋은 baseline
- **Transformer는 위치 쌍을 직접 연결한다**
  ```
  RNN:            1 → 2 → 3 → ... → L      경로 O(L)
  Self-Attention: 1 ──────────────→ L      한 층 경로 O(1)
  ```
  - 한 층의 모든 위치를 큰 행렬 연산으로 계산 → 학습 시 **sequence 축 병렬화**가 쉬움
  - 대신 기본 score 행렬이 `[L, L]`이라 원소 수가 `L²`로 증가
    ```
    L=4   → RNN path 3    / Attention scores 16
    L=16  → RNN path 15   / Attention scores 256
    L=128 → RNN path 127  / Attention scores 16,384
    ```
  - ⭐ 핵심: Transformer의 장점은 "모든 관계를 공짜로 본다"가 아니라 **정보 경로를 짧게 만드는 대신 `L²` score 비용을 부담한다**는 것. `O(L²)`은 *경로*가 아니라 *score 행렬의 원소 수*
  - ⛔ 학습의 병렬성과 생성의 순차성은 모순이 아님 — 학습에서는 정답 token이 주어져 여러 위치 logits을 한 번에 계산하지만, **추론 생성에서는 다음 token이 직전까지 생성된 token에 의존**
- 💡 **강의 밖 보강 개념**
  - Self-Attention은 `softmax(QKᵀ / √d_k)·V` — `√d_k`로 나누는 이유는 내적이 커지면 softmax가 한쪽으로 포화돼 gradient가 죽기 때문
  - Attention 연산 자체는 **순서를 모름(permutation invariant)** → 그래서 **Positional Encoding/Embedding이 필수**. RNN은 구조 자체가 순서를 담고 있어 필요 없었음
  - 층 단위 비용은 Self-Attention `O(L²·d)` vs RNN `O(L·d²)` → **`L`이 짧고 `d`가 크면 Attention이 오히려 저렴**, `L`이 아주 길어질 때 비싸짐
  - 생성 시에는 **KV cache**로 이전 위치의 K·V를 재사용해 매 step 전체를 다시 계산하지 않음
  - ⛔ 양방향 RNN 출력은 미래 입력을 보므로 **실시간 예측처럼 미래를 볼 수 없는 문제에 그대로 적용 불가**
- **비교표: 우열이 아니라 제약의 차이**

  | 관점 | RNN/LSTM | Transformer |
  | --- | --- | --- |
  | 같은 layer 안 정보 경로 | 위치 수에 따라 길어짐 | 먼 위치도 직접 연결 |
  | 학습 시 시간축 계산 | 순차 의존 | 행렬 단위 병렬 |
  | 긴 문맥 | 상태에 계속 축약 | 모든 위치를 직접 비교 |
  | 주요 비용 | 긴 직렬 경로 | 기본 Attention의 `L²` 메모리·연산 |
  | 유리한 조건 | 스트리밍, 작은 모델, 짧은 시계열 | 대규모 학습, 긴 문맥 관계, 전이학습 |

- 📝 **이해도 점검**
  1. `[B, L, H]`인 RNN `output`과 `[layers×directions, B, H]`인 `h_n`은 각각 무엇인가?
     → `output`은 **모든 시점의 표현**, `h_n`은 **층·방향별 마지막 상태**. 다층·양방향에서 둘을 같은 값으로 단정하면 안 됨
  2. Gradient clipping이 해결하는 문제와 해결하지 못하는 문제는?
     → 폭주하는 **gradient 크기는 제한**하지만, **장거리 관계 학습과 시간축 순차 계산 자체는 없애지 못함**
  3. 길이 128에서 첫 위치와 마지막 위치 사이의 RNN 경로 수와 Self-Attention score 원소 수는?
     → 경로는 **127**, score 원소 수는 **16,384**(= 128²)
  4. Transformer 학습의 병렬성과 생성의 순차성이 모순이 아닌 이유는?
     → 학습에서는 정답 token이 주어져 여러 위치 logits을 함께 계산할 수 있지만, **추론 생성에서는 다음 token이 직전까지 생성된 token에 의존**하기 때문

**1장2강. 미니 LLM 워크플로우와 누적 산출물**
- 도구 이름이 바뀌어도 유지되는 **실험 계약(순서)**
  ```
  문제 정의 → 데이터 감사·split → tokenize·collate → model·objective
  → train → validation으로 선택 → test 1회 평가 → 저장·재로드
  → 새 입력 inference → error analysis → 최종 실험 기록
  ```
  - 먼저 "무엇을 예측하고 어떤 기준으로 성공이라 할지"를 정하고, 마지막에는 **다른 사람이 같은 환경·설정으로 재현할 수 있게** 기록
- **raw text가 모델 입력이 되는 경로**
  ```
  raw text ─ tokenizer ─┬ input_ids       [B, L]
                        └ attention_mask  [B, L]  ─┐
  분류 정답 ─ label mapping ─ labels        [B]    ─┼→ model → loss / logits
  LM input_ids ─ collator/objective ─ labels [B, L]─┘
  ```

  | 필드 | 의미 |
  | --- | --- |
  | `input_ids` | vocabulary의 정수 ID |
  | `attention_mask` | 실제 token은 1, PAD 위치는 0 |
  | `labels` | **tokenizer의 산출물이 아님** — Dataset의 분류 정답이거나 LM collator·objective가 만드는 학습 목표 |
  | `logits` | 정규화 전 점수 |
  | `loss` | 학습할 때 gradient를 만드는 목적 함수 |

  - ⛔ tokenizer와 model은 **같은 checkpoint 계열**이어야 함 — ID는 tokenizer의 vocabulary에 종속되기 때문
- **Task별 출력 계약** — 같은 `AutoTokenizer` 입구를 써도 head·출력·평가는 다름

  | Task | 대표 model class | Logits shape | 대표 평가 |
  | --- | --- | --- | --- |
  | 문장 분류 | `AutoModelForSequenceClassification` | `[B, C]` | accuracy, macro-F1 |
  | Masked LM | `AutoModelForMaskedLM` | `[B, L, V]` | mask 위치 top-k |
  | Causal LM | `AutoModelForCausalLM` | `[B, L, V]` | 생성 품질, 지연, 안전성 |

  ```python
  probabilities = torch.softmax(logits, dim=-1)   # 클래스 축 C에만 softmax → 행 합이 1
  predictions   = logits.argmax(dim=-1)           # 각 샘플에서 가장 큰 클래스 ID
  ```
  - ⛔ Softmax는 **클래스 축 `dim=-1`**에 적용 — batch 축에 적용하면 서로 다른 샘플이 서로 경쟁하는 잘못된 계산이 됨
  - 💡 보강: logit은 확률이 아니라 실수 점수라 합이 1일 필요가 없고, `CrossEntropyLoss`는 내부에서 log_softmax를 하므로 **모델 출력에 softmax를 또 씌우면 안 됨**. `argmax`는 softmax 전후가 동일
- **학습·평가·추론의 역할 구분**

  | 단계 | 정답 label | 파라미터 갱신 | 목적 |
  | --- | --- | --- | --- |
  | Train | 있음 | 있음 | loss를 줄이도록 학습 |
  | Validation | 있음 | 없음 | hyperparameter·checkpoint 선택 |
  | Test | 있음 | 없음 | 모든 선택 후 최종 일반화 성능 확인 |
  | Inference | 보통 없음 | 없음 | 새 입력에 실제 결과 생성 |

  - ⛔ test 결과를 보고 epoch이나 설정을 다시 고르면 **test가 validation 역할**을 하게 되어 성능이 낙관적으로 편향됨 → 최종 기록에 **"test를 몇 번 확인했는지"**를 남김
- **생성에서 `forward`와 `generate()`는 다르다**
  ```
  forward:  input → [B, L, V] logits              (한 번의 계산)
  generate: forward → token 선택 → append → forward → ... → decode  (종료 조건까지 반복)
  ```
  - `max_new_tokens`는 **새로 만들 token 수만**, `max_length`는 입력과 출력을 합친 **전체 길이**를 제한
- **재현 가능한 실행 기록(manifest)** — 최소 단위
  ```python
  manifest = {
      "problem": "한국어 뉴스 제목 3분류",
      "dataset_version": "news-mini-v1",
      "split_seed": 42,                     # 같은 split을 다시 만들기 위한 seed
      "model_id": "monologg/koelectra-small-v3-discriminator",
      "tokenizer_id": "monologg/koelectra-small-v3-discriminator",
      "max_length": 64,                     # 입력 길이 분포를 본 뒤 고른 절단 상한
      "selection_metric": "validation_macro_f1",
      "test_policy": "best checkpoint 확정 후 1회",
  }
  ```
  - ⛔ **seed만으로는 부족** — package version, dataset fingerprint, model revision, 실제 checkpoint 경로와 평가 코드까지 함께 기록
  - 표와 로그에서는 **"실제 실행값"과 "해석"을 분리**하고, 예상값을 실행값처럼 쓰지 않음
- **오류·주의사항**
  - `model_id`와 tokenizer의 checkpoint를 다르게 섞지 않기
  - accuracy와 loss를 같은 개념으로 설명하지 않기 (loss는 최적화 대상, accuracy는 결과 지표)
  - **마지막 checkpoint를 자동으로 best checkpoint라고 가정하지 않기**
  - 생성 문자열이 자연스럽다는 이유만으로 사실성이 검증됐다고 결론 내리지 않기

- 📝 **이해도 점검**
  1. 분류 logits `[8, 3]`의 두 축은?
     → **샘플 8개와 클래스 3개**
  2. Validation과 test를 나누는 이유는?
     → **선택에 사용한 데이터**와 **최종 평가 데이터**를 분리해 낙관적 편향을 줄이기 위해
  3. Model과 tokenizer를 함께 저장해야 하는 이유는?
     → 같은 문자열을 **같은 ID와 special token 규칙으로 재현**해야 하기 때문
  4. `generate()`와 forward 한 번의 차이는?
     → forward는 logits를 **한 번 계산**, `generate()`는 **token 선택 → append → forward를 반복**

**2장1강. NLP 데이터 구조와 Token·Subword**
- ⭐ **Tokenizer보다 데이터 계약이 먼저** — 원문과 라벨을 믿을 수 없으면 그 뒤의 token ID와 metric도 믿을 수 없음
  ```python
  # 각 dict가 데이터셋의 한 행: id는 중복 추적, text는 모델 입력, label은 사람이 읽는 클래스 이름
  required = {"id", "text", "label"}
  for row in samples:
      assert required <= row.keys()                              # 필수 열 누락을 가장 먼저 차단
      assert row["id"] and row["text"].strip() and row["label"]  # 빈 값 방어
  ```
  - 빈 문자열, 중복 ID, 예상 밖 label, 동일 텍스트의 split 간 중복을 **tokenization 이전에** 확인 — 이후에는 사람이 읽는 원문에서 오류를 찾기가 훨씬 어려움
- **용어 정리**

  | 용어 | 뜻 |
  | --- | --- |
  | Token | tokenizer가 모델 입력의 기본 단위로 취급하는 조각 |
  | Vocabulary | token 문자열 ↔ 정수 ID의 **고정 대응표** |
  | Token ID | vocabulary에서 해당 token을 가리키는 정수 |
  | UNK/OOV | vocabulary로 표현하지 못한 입력 또는 미등록 단위 |

  - ⛔ **ID 숫자 자체에는 보편적 의미가 없음** — `1234`가 어떤 token인지는 특정 tokenizer와 vocabulary 버전에 따라 달라짐
- **왜 subword를 쓰는가** — word와 character의 중간 절충

  | 단위 | 장점 | 비용·한계 |
  | --- | --- | --- |
  | Word | 사람이 읽기 쉽고 sequence가 짧음 | 어휘가 매우 커지고 신조어/OOV가 많음 |
  | Character | 작은 어휘로 거의 모든 문자열 표현 | sequence가 길어지고 의미 단위가 잘게 깨짐 |
  | Subword | 재사용되는 조각으로 **OOV와 길이의 균형** | 분할이 직관적 단어 경계와 다를 수 있음 |

  - 예: `인공지능연구소`가 하나의 word token으로 없더라도 `인공`, `##지능`, `연구`, `##소`처럼 **기존 조각을 조합**해 표현
  - 전체 흐름: `문자열 → 정규화·pre-tokenization·subword 알고리즘 → token 문자열 → vocabulary lookup → input_ids`
  - 💡 보강: subword 알고리즘 계열의 차이
    - **BPE** — 빈도가 높은 인접 쌍을 반복 병합해 vocabulary 구성 (GPT 계열)
    - **WordPiece** — 병합 기준이 단순 빈도가 아니라 likelihood 증가량, 이어지는 조각을 `##`로 표기 (BERT·KoELECTRA 계열)
    - **Unigram / SentencePiece** — 후보 집합에서 확률이 낮은 token을 제거해 감축, 공백까지 기호(`▁`)로 다뤄 언어 독립적
- **Vocabulary 크기와 sequence 길이의 trade-off**
  - vocabulary ↑ → 자주 쓰는 긴 문자열을 한 token에 담아 **길이는 짧아지지만 embedding·LM head 파라미터가 늘어남**
  - vocabulary ↓ → 희귀 문자열을 더 작은 조각으로 표현해 **sequence가 길어짐**
  - ⛔ 이는 단순한 경향일 뿐 — tokenizer algorithm, 학습 corpus, 정규화 규칙이 함께 영향을 주므로 **vocabulary size 하나만으로 tokenizer 품질을 평가하지 않음**
- **Special token은 모델 입력의 구조를 표시한다**

  | Token 역할 | 대표 표기 | 용도 |
  | --- | --- | --- |
  | Padding | `[PAD]` | batch 길이 맞춤 |
  | Unknown | `[UNK]` | 표현하지 못한 문자열 |
  | Classification/Beginning | `[CLS]`, `<s>` | 문장 대표 또는 시작 |
  | Separator/End | `[SEP]`, `</s>` | 문장 경계 또는 끝 |
  | Mask | `[MASK]` | Masked LM 학습·추론 |

  - ⛔ 모든 모델이 같은 표기와 역할을 쓰지 않음 → **Token ID를 하드코딩하지 말고** `tokenizer.pad_token_id`, `tokenizer.mask_token`처럼 **객체에서 읽기**
- **출력 예시를 읽는 방법** — 특정 ID 숫자가 아니라 *계약*을 본다
  ```
  text:   인공지능연구소 출범
  tokens: [CLS], 인공, ##지능, 연구, ##소, 출범, [SEP]
  ids:      2,   4102,  7821,  991,  441, 7300,   3
  ```
  - token 7개와 ID 7개가 **1:1 대응**, `##`는 앞 조각에 이어지는 subword, 문장 경계 special token이 앞뒤에 추가됨
  - ⛔ 같은 원문의 token 수가 A=7, B=11이라고 해서 A가 무조건 우수한 것은 아님 — **downstream 모델 호환성, 희귀 도메인 용어 처리, 전체 길이 분포, `[UNK]` 비율**을 함께 확인
- ⭐ **label ID와 token ID는 서로 다른 namespace**
  ```python
  label2id = {"economy": 0, "sports": 1, "tech": 2}   # task가 정하는 별도 namespace
  id2label = {v: k for k, v in label2id.items()}      # 예측 ID를 사람이 읽는 이름으로 복원

  assert len(label2id) == len(set(label2id.values()))                # 서로 다른 label이 같은 ID 공유 금지
  assert all(id2label[label2id[name]] == name for name in label2id)  # 왕복 변환 검사
  ```
  - 모델 예측의 `0`은 economy label일 수 있지만 `input_ids`의 `0`은 PAD 같은 **전혀 다른 token**일 수 있음 → 어느 namespace의 ID인지 변수명과 표 제목에 명시
- **데이터 누수·중복 점검 6가지**: ① ID 유일성 ② 텍스트 정규화 후 중복 ③ label 집합과 `label2id` 일치 ④ 클래스 분포 ⑤ 출처·시간 단위 split 필요성 ⑥ 개인정보·라이선스·민감 정보
  - 같은 기사 제목이 train과 test에 동시에 있으면 모델이 일반화한 것이 아니라 **본 문장을 기억한 효과가 점수에 반영**됨
- **오류·주의사항**: "token = 단어"로 고정하지 않기 / `len(text)`는 **문자 수**이지 token 수가 아님 / `[UNK]`가 적다고 무조건 좋은 tokenizer는 아님 / 원문 label 문자열과 학습용 정수 ID 매핑을 **양방향으로 저장**

- 📝 **이해도 점검**
  1. Subword가 word와 character의 중간 절충인 이유는?
     → word보다 작은 조각을 재사용해 **OOV를 줄이면서**, character보다 **sequence를 덜 늘리기** 때문
  2. 같은 문자열의 token ID가 tokenizer마다 달라질 수 있는 이유는?
     → 정규화·분할 알고리즘·vocabulary가 **checkpoint마다 다르기** 때문
  3. `label2id`와 `id2label`을 모두 저장해야 하는 이유는?
     → **학습 입력 변환**(이름 → ID)과 **사람이 읽는 예측 복원**(ID → 이름)을 모두 재현하기 위해
  4. Split 간 중복이 평가를 왜곡하는 과정은?
     → test 문장이 train에도 존재하면 **기억 효과가 일반화 성능처럼** 점수에 반영됨

**2장2강. AutoTokenizer encode/decode와 입력 필드**
- ⭐ `AutoTokenizer.from_pretrained(model_id)`는 **checkpoint의 `tokenizer_config`를 읽어 적합한 tokenizer 구현을 선택하는 config 기반 factory** — `Auto`는 임의의 tokenizer를 섞어 주는 기능이 아님
  ```python
  from transformers import AutoTokenizer

  MODEL_ID = "monologg/koelectra-small-v3-discriminator"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

  print(type(tokenizer).__name__, tokenizer.vocab_size, tokenizer.special_tokens_map)
  ```
  - 운영 실험에서는 **revision을 commit SHA로 고정**하고 package version도 함께 고정. 캐시만 사용할 때는 `local_files_only=True`
- **다섯 API의 경계를 구분한다**

  | API | 대표 반환 | 주요 용도 |
  | --- | --- | --- |
  | `tokenize(text)` | token 문자열 list | 분할 결과 **관찰** |
  | `convert_tokens_to_ids(tokens)` | ID list | token ↔ ID 대응 확인 |
  | `encode(text)` | ID list | 한 문장을 간단히 ID로 변환 |
  | `tokenizer(texts, ...)` | `BatchEncoding` | **실제 model 입력 생성** |
  | `decode(ids)` | 문자열 | ID를 읽을 수 있는 텍스트로 복원 |

  - ⭐ 판단 기준: **"지금 사람이 분할을 관찰하는가, 아니면 모델에 넣을 batch를 만드는가"**
  - tokenizer 내부 파이프라인: `정규화 → pre-tokenization → tokenization 모델 → post-processing → decoding` — **special token과 입력 필드는 post-processing 단계에서 추가**됨
- **한 문장을 끝까지 추적하기**
  ```python
  text = "한국어 모델은 문장을 토큰으로 나눕니다."

  tokens           = tokenizer.tokenize(text)                          # 사람이 읽기 위한 token 문자열
  ids_without      = tokenizer.convert_tokens_to_ids(tokens)           # 경계 token 없는 ID
  ids_with_special = tokenizer.encode(text, add_special_tokens=True)   # 모델이 기대하는 [CLS]/[SEP] 포함
  decoded          = tokenizer.decode(ids_with_special, skip_special_tokens=True)
  ```
  - ⛔ `decode` 결과는 원문과 **공백·정규화가 정확히 같지 않을 수 있음** → 문자열 완전 일치를 universal contract로 두지 말고 **의미와 token sequence**로 확인
- **`BatchEncoding`과 shape**
  ```python
  batch = tokenizer(
      texts,
      padding=True,          # 이 batch의 최장 문장까지만 PAD 추가
      truncation=True,       # max_length를 넘는 입력은 명시적으로 절단
      max_length=16,         # 실제 값은 길이 분포로 결정
      return_tensors="pt",   # list가 아닌 PyTorch Tensor [B, L] 반환
  )
  ```
  - ⭐ `input_ids`와 `attention_mask`는 보통 `[B, L]` — **`B`는 문장 수, `L`은 이 batch에서 padding 후 token 길이(단, `max_length` 이하)**. hidden 차원이 아님
  - KoELECTRA/BERT 계열은 `token_type_ids`도 반환 — **모델·tokenizer에 따라 없을 수 있으므로 무조건 가정하지 않고** 반환된 key를 순회하거나 `if "token_type_ids" in batch`로 조건 처리
  - 💡 보강: `token_type_ids`는 문장 쌍 입력(`[CLS] A [SEP] B [SEP]`)에서 첫 문장 0 / 둘째 문장 1로 구분하는 **segment 정보**
- **Special token은 tokenizer가 넣게 한다**
  - `add_special_tokens=True`가 기본인 호출에서 `[CLS]`, `[SEP]` 같은 **모델별 경계 token**이 자동 추가됨
  - ⛔ 문자열로 직접 붙이면 **중복 special token이나 잘못된 ID**가 생길 수 있음
  ```python
  ids    = batch["input_ids"][0]
  tokens = tokenizer.convert_ids_to_tokens(ids)
  # 각 token 옆에 1(실제 입력) 또는 0(PAD)을 붙여 눈으로 확인
  print(list(zip(tokens, batch["attention_mask"][0].tolist())))
  ```
  - PAD 위치까지 보고 싶으면 `decode(..., skip_special_tokens=False)`, 사람에게 보여줄 최종 텍스트는 보통 `skip_special_tokens=True`
- **출력 계약을 검증하는 assert** — 조용한 전처리 오류를 줄이는 장치
  ```python
  assert batch["input_ids"].shape == batch["attention_mask"].shape   # 같은 [B, L] 격자를 설명해야 함
  assert batch["input_ids"].dtype == torch.long                      # embedding index는 정수형
  assert batch["attention_mask"].max().item() <= 1                   # 대표적인 binary mask 계약
  assert batch["attention_mask"].min().item() >= 0
  ```
  - **필드 이름·shape·dtype·special token ID를 함께 검증**해야 눈에 띄지 않는 전처리 오류를 줄일 수 있음
- **오류·주의사항**: 다른 tokenizer의 token ID를 그대로 모델에 전달하지 않기 / `decode`가 원문 공백까지 완전 복원한다고 가정하지 않기 / `BatchEncoding`의 key가 모델마다 동일하다고 하드코딩하지 않기 / `return_tensors="pt"` 없이 `.shape`·`.to(device)`를 바로 쓰지 않기 / 한국어 터미널 표시 인코딩이 깨진 것과 실제 ID·shape 오류를 구분하기

- 📝 **이해도 점검**
  1. `tokenize()`와 `tokenizer(...)`의 목적 차이는?
     → 전자는 **분할 관찰용 token 문자열**, 후자는 **batch와 mask를 포함한 실제 모델 입력 생성**에 적합
  2. `input_ids [4, 12]`의 두 축은?
     → **문장 4개**와 **padding 후 길이 12**
  3. `token_type_ids`가 없는 모델에서도 코드가 동작하게 하려면?
     → 반환된 key를 순회하거나 `if "token_type_ids" in batch`로 **조건 처리**
  4. Model과 tokenizer checkpoint를 맞춰야 하는 이유는?
     → 같은 ID가 **같은 token과 special token 규칙**을 가리켜야 하기 때문

**2장3강. Padding·Truncation·Attention Mask와 Dataset 파이프라인**
- **셋은 서로 다른 문제를 푼다**
  - **Padding**: 짧은 입력을 PAD token으로 채워 한 batch를 **직사각형 Tensor**로 만듦 → 정보 손실은 없지만 **계산 낭비**가 늘 수 있음
  - **Truncation**: 모델·정책의 최대 길이를 넘는 token을 자름 → 계산량은 제한되지만 **중요한 뒤 문맥을 잃을 수 있음**
  - **Attention mask**: 실제 key와 PAD key를 구분
  ```
  원래 길이:      [7, 4, 5]
  동적 padding:   [7, 7, 7]   ← 현재 batch 최장 길이
  max_length=6:   [6, 4, 5]   ← 긴 입력은 먼저 절단
  최종 batch:     [6, 6, 6]
  ```
  - ⭐ 적용 순서는 **truncation이 먼저, padding이 나중**
  - 💡 보강: attention mask는 단순 표시가 아니라 score 행렬의 PAD 열에 매우 큰 음수(`-inf`)를 더해 **softmax 결과를 0으로 만드는** 방식으로 동작
- **정적 padding과 동적 padding**

  | 방식 | 설정 위치 | 장점 | 주의점 |
  | --- | --- | --- | --- |
  | 정적 | tokenizer의 `padding="max_length"` | shape 고정, 단순 | 짧은 batch에도 PAD가 많음 |
  | 동적 | collator의 `padding=True` | batch별 낭비 감소 | batch마다 `L`이 달라짐 |

  - PyTorch BetterTransformer 실측: batch 안 **padding 비율이 커질수록** 불필요한 PAD 계산을 건너뛰는 최적화 효과가 커짐 → **"shape를 맞추기 위한 PAD도 실제 계산 비용을 낸다"**는 근거
  - 💡 보강: `pad_to_multiple_of=8`로 텐서코어 정렬을 맞추거나, 길이가 비슷한 샘플끼리 묶는 **length grouping**을 함께 쓰면 PAD 낭비를 더 줄일 수 있음
- ⭐ **실행 순서: split은 먼저, padding은 batch 직전에**
  ```
  원본 Dataset(길이 제각각) → 먼저 Split(train/validation/test)
      → map(tokenize, 아직 padding 안 함) → Data Collator(그 batch의 최장 길이로 동적 padding)
  ```
  - split을 먼저 고정해야 **같은 문장이 train과 test에 섞이는 일**을 막을 수 있음
  ```python
  splits = DatasetDict({
      "train":      dataset.select([0, 1, 3, 4, 6, 7]),
      "validation": dataset.select([2, 5, 8]),
      "test":       dataset.select([9, 10, 11]),
  })
  # 같은 원본 행이 두 split에 들어가지 않았는지 index 수준에서도 확인
  assert split_indices["train"].isdisjoint(split_indices["validation"])
  assert split_indices["train"].isdisjoint(split_indices["test"])
  assert split_indices["validation"].isdisjoint(split_indices["test"])
  ```
  - 실제 프로젝트에서는 **고정 sample ID를 별도 열로 보존**하고 split 간 교집합이 비었는지 자동 검사
- **`map(batched=True)`로 세 split에 같은 함수·같은 config 적용**
  ```python
  def tokenize_batch(batch):
      # batched=True에서는 batch["text"]가 문자열 하나가 아니라 list
      encoded = tokenizer(
          batch["text"],
          truncation=True,
          max_length=16,
          padding=False,        # ⭐ map 단계에서는 원래 token 길이를 보존
      )
      encoded["labels"] = batch["label"]   # model이 기대하는 복수형 key
      return encoded

  tokenized = splits.map(tokenize_batch, batched=True)
  ```
  - `batched=True`면 함수는 한 행이 아니라 **열별 list 묶음**을 받고, 반환 list의 길이는 입력 batch 행 수와 같아야 함
  - ⛔ 모델·Trainer가 기대하는 key는 `label`이 아니라 **`labels`**인 경우가 많음
- **Collator가 실제 batch를 만든다**
  ```python
  collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

  # 원문 text·label 문자열은 오류 분석용으로 Dataset에 남기고, Collator에는 숫자 열만 전달
  model_input_keys = ["input_ids", "attention_mask", "labels"]
  if "token_type_ids" in tokenized["train"].column_names:
      model_input_keys.append("token_type_ids")

  batch = collator(features)     # 이 호출 시점에 features 중 최장 길이까지만 PAD

  pad_positions = batch["input_ids"].eq(tokenizer.pad_token_id)
  # PAD ID가 있는 모든 칸과 attention_mask=0인 칸이 정확히 같아야 함
  assert torch.equal(pad_positions, batch["attention_mask"].eq(0))
  ```
  - ⭐ **PAD ID 위치와 `attention_mask=0` 위치의 일치**가 tokenizer/model 입력 계약의 핵심 — 어긋나면 모델이 **PAD를 문맥으로 읽거나 반대로 실제 token을 가려버림**
  - 💡 보강: LM 학습에는 `DataCollatorForLanguageModeling`을 쓰고, loss에서 제외할 위치는 label을 `-100`으로 채움(`CrossEntropyLoss`의 기본 `ignore_index`). decoder-only 모델 생성 시에는 보통 **left padding**을 사용
- **최대 길이는 감이 아니라 데이터로 정한다**
  ```python
  # max_length 정책을 정할 때 test 입력을 들여다보지 않고 train 길이부터 측정
  lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in splits["train"]["text"]]
  for candidate in (8, 16, 32):
      cut = sum(length > candidate for length in lengths)      # 잘릴 샘플 수
      print(candidate, cut, f"{cut / len(lengths):.1%}")
  ```
  - 평균만 보면 긴 꼬리를 놓치므로 **median, p95, max와 후보별 절단 비율**을 함께 확인
  - ⛔ 최종 길이는 **validation 성능·메모리와 함께 결정**하고, test 절단률은 정책을 바꾸기 위한 입력이 아니라 **확정된 정책의 적용 결과로만 기록**. `max_length`를 키우면 메모리와 계산량이 증가
- **Split과 전처리에서 지켜야 할 경계 5가지**
  1. 원본을 먼저 감사하고 split ID를 고정
  2. **학습 데이터에만 fitting이 필요한 변환**(vocabulary·통계)은 train에서만 학습
  3. Tokenizer처럼 **고정된 사전학습 변환**은 세 split에 같은 config로 적용
  4. Validation으로 길이·학습 설정을 결정
  5. Test는 선택 완료 후 **한 번** 평가
- **오류·주의사항**: `max_length`만 쓰고 `truncation=True`를 빠뜨리지 않기 / Dataset 전체를 처음부터 최대 길이로 padding하지 않기 / `label`과 `labels` 이름이 Trainer·model contract에서 어떻게 쓰이는지 확인 / `remove_columns`로 원문을 없애기 전에 error analysis에 필요한 ID·text 보존 정책 정하기 / collator에 이미 Tensor인 서로 다른 길이 자료를 넘겨 수동 stack 오류를 만들지 않기

- 📝 **이해도 점검**
  1. Dataset에서는 `padding=False`, collator에서는 padding하는 이유는?
     → 가변 길이를 보존하다 **실제 batch의 최장 길이에만 맞춰** PAD 낭비를 줄이기 위해
  2. `input_ids [6, 14]`의 `14`는 전체 데이터의 최대 길이인가?
     → 아님. 동적 padding이면 **현재 6개 샘플 batch의 최대 길이**
  3. PAD 위치의 attention mask가 0이어야 하는 이유는?
     → 모델이 **채움 token을 실제 문맥 key로 참고하지 않게** 하기 위해
  4. Truncation 비율이 0%여도 `max_length` 기록이 필요한 이유는?
     → 재실행 시 **같은 입력 정책을 보장**하고, **더 긴 새 데이터가 들어올 때의 동작을 설명**하기 위해

**오늘 배운 것 한눈에 정리**
- RNN은 hidden state로 순서를 반영하지만 **`h_t`가 `h_{t-1}`에 의존**해 긴 직렬 경로를 만들고, LSTM은 cell state와 gate로 gradient·기억 문제를 완화할 뿐 **시간축 순차성은 그대로** 남음
- Self-Attention은 먼 위치를 **한 층에서 `O(1)` 경로로 직접 연결**하고 학습 시 행렬 병렬화를 가능하게 하지만, 그 대가로 `[L, L]` score 행렬의 **`L²` 메모리·연산 비용**을 부담 — 우열이 아니라 제약의 차이
- LLM 실험은 **문제 정의 → split → tokenize·collate → model·objective → train → validation 선택 → test 1회 → 저장·추론·오류 분석 → 기록**의 계약을 따르고, seed뿐 아니라 **version·revision·설정·산출물 경로**까지 남겨야 재현 가능
- 분류와 생성은 tokenizer 입구를 공유해도 **head, logits shape(`[B, C]` vs `[B, L, V]`), metric, 후처리가 다름**
- Token은 단어가 아니며 **subword는 OOV·vocabulary 크기·sequence 길이 사이의 절충** — vocabulary size 하나로 tokenizer 품질을 판단하지 않음
- **label ID와 token ID는 다른 namespace**이므로 양방향 매핑(`label2id` / `id2label`)과 checkpoint를 각각 기록
- `AutoTokenizer`는 checkpoint config에 맞는 구현을 고르는 **factory**, `tokenize()`는 관찰용, `tokenizer(...)`는 batch·mask를 포함한 **실제 모델 입력 생성용**
- Dataset `map`에서는 `padding=False`로 **가변 길이를 보존**하고, `DataCollatorWithPadding`이 **현재 batch 최장 길이까지만** 동적 padding — PAD 위치와 `attention_mask=0`은 반드시 일치해야 함

**데일리 퀴즈 정리**
1. `AutoTokenizer.from_pretrained(model_id)`가 수행하는 역할 → **checkpoint의 `tokenizer_config`를 읽어 적합한 tokenizer 구현을 선택하는 factory**
   - `Auto`는 임의의 tokenizer를 섞는 기능이 아니라 **config 기반 factory**
   - 그래서 model과 tokenizer는 같은 checkpoint 계열로 맞춰야 하고, 운영에서는 revision까지 고정하는 편이 안전
2. `tokenizer(...)` 결과 `batch["input_ids"]`의 shape가 `[4, 12]`일 때 두 축의 의미 → **`B` = 문장 수 4, `L` = 해당 batch의 padding 후 token 길이 12**
   - `input_ids`와 `attention_mask`는 대표적으로 `[B, L]` 형태
   - ⛔ 두 번째 축을 hidden state 차원으로 읽으면 안 됨 — hidden 차원은 모델 내부 표현(`[B, L, H]`)에서 등장하고 **tokenizer 출력에는 아직 hidden state가 없음**
   - 같은 데이터라도 **동적 padding이면 batch마다 `L`이 달라짐**
3. 길이 `L`인 시퀀스에서 첫 위치와 마지막 위치 사이의 정보 경로 → **RNN은 `O(L)`의 순차 경로, Self-Attention은 한 층에서 `O(1)` 경로로 직접 연결**
   - RNN 경로는 `1→2→...→L`이고, Self-Attention은 한 층에서 두 위치를 바로 연결
   - ⛔ `O(L²)`은 **경로가 아니라 `[L, L]` score 행렬의 원소 수** — Attention이 그 대신 지불하는 메모리·연산 비용에 해당
4. Dataset의 `map` 단계에서 `padding=False`로 설정하는 이유 → **원래 token 길이를 보존해 두고, batch를 만들 때 필요한 만큼만 padding해 메모리·계산을 아끼기 위해**
   - `DataCollatorWithPadding`이 **현재 batch의 최장 길이까지만** 동적으로 padding
   - 전체를 미리 최대 길이로 채우면 짧은 batch에서도 PAD 계산 비용을 그대로 지불하게 됨
5. Validation과 Test를 별도의 split으로 나누어 사용하는 이유 → **hyperparameter·checkpoint 선택에는 validation을, 모든 선택이 끝난 뒤 최종 일반화 성능 확인에는 test를 사용하기 위해**
   - ⛔ test를 보고 다시 설정을 고르면 **test가 validation 역할**을 하게 되어 성능이 낙관적으로 편향됨
   - "validation은 모델 저장용, test는 추론 서비스용"처럼 **용도(저장/서비스)로 나누는 구분이 아님** — 둘 다 파라미터를 갱신하지 않는 평가 단계이고, 차이는 **선택에 쓰였는지 여부**
   - 그래서 최종 실험 기록에는 **test를 몇 번 확인했는지**까지 남김

**한 줄 정리**: RNN·LSTM은 hidden state를 한 단계씩 넘기며 순서를 반영하지만 첫 위치에서 마지막 위치까지 `O(L)`의 직렬 경로와 시간축 순차성이 남고(LSTM의 gate·cell state는 gradient 흐름만 완화할 뿐), Self-Attention은 위치 쌍을 한 층에서 `O(1)`로 직접 연결하는 대신 `[L, L]` score 행렬의 `L²` 비용을 지불하는 **제약의 교환**이며 — 실제 LLM 실험은 문제 정의부터 split·tokenize·collate·학습·validation 선택·test 1회 평가·기록까지 이어지는 계약 위에서 돌아가고, 그 입구인 tokenizer는 checkpoint config에 맞는 구현을 고르는 `AutoTokenizer` factory로 불러와 `tokenizer(...)`가 `[B, L]`의 `input_ids`·`attention_mask`를 만들며, token은 단어가 아니라 OOV와 길이를 절충한 subword이고 label ID와 token ID는 서로 다른 namespace이므로 양방향 매핑을 남겨야 하고, 길이 처리는 **split을 먼저 고정 → `map`에서 `padding=False`로 가변 길이 보존 → collator가 현재 batch 최장 길이까지만 동적 padding**의 순서로 나누되 PAD 위치와 `attention_mask=0`이 정확히 일치하는지 검증하고 `max_length`는 train 길이 분포와 validation 성능으로 정해야 한다.
