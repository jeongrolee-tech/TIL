# 2026-09-07 BERT·GPT 구조 비교(Masked LM과 Causal LM) 그리고 Hugging Face Pipeline·AutoModel 활용

## 오늘의 TIL 요약

**5장1강. 사전학습 언어모델과 LM Objective**
- **언어모델이 학습하는 것** — 사전이 아니라 **패턴**
  - 토큰의 **등장 순서** (무엇 다음에 무엇이 오는가)
  - 함께 등장하는 **공기(co-occurrence) 관계**
  - **문맥에 따른 의미 변화** (같은 표기라도 주변에 따라 뜻이 달라짐)
- **사전학습(Pre-training) vs Fine-tuning**

  | 구분 | 데이터 | 목적 | 비용 |
  | --- | --- | --- | --- |
  | Pre-training | 대규모 **비라벨** 텍스트 | 재사용 가능한 일반 언어 패턴 학습 | 매우 큼 |
  | Fine-tuning | 소규모 **목적별 라벨** 데이터 | 특정 문제에 맞게 추가 학습 | 상대적으로 작음 |

  - ⭐ 실무에서 직접 하는 일은 대부분 **Fine-tuning 또는 그대로 추론**이며, 사전학습은 이미 태워진 연료를 쓰는 쪽에 가까움
- **LM Objective = 모델이 사전학습 중 반복해서 푸는 문제**
  - "어떤 구조를 쓰는가(Architecture)"와 "어떤 문제를 푸는가(Objective)"는 **서로 다른 축**

  | Objective | 대표 질문 | 참고하는 문맥 | 대표 모델 |
  | --- | --- | --- | --- |
  | Masked LM | 가려진 자리에 원래 있던 토큰은? | 왼쪽 + 오른쪽 | BERT 계열 |
  | Causal LM | 지금까지의 문맥 다음에 올 토큰은? | 왼쪽 + 현재 | GPT 계열 |

  - 💡 구조와 Objective를 따로 봐야 하는 이유 — **같은 구조라도 다른 Objective를 붙일 수 있음**. 구조는 *정보 흐름*을, Objective는 *학습 문제*를 설명함
- **Representation Learning** — 문맥이 반영된 내부 표현을 잘 만드는 쪽
  - 예: "**배**가 아파서 병원에 갔다" / "**배**를 타고 섬에 들어갔다" → 표기는 같지만 필요한 표현이 다름
  - Encoder 계열은 각 토큰이 문장 전체를 참고하게 만들어 **표현 자체를 문맥에 맞게 조정**
  - 활용: 문장 분류, 토큰 분류(NER·개인정보 탐지), 검색 임베딩, 유사도, QA용 문맥 표현
- **Generation Learning** — 주어진 문맥 뒤를 순차적으로 이어 쓰는 쪽
  ```
  Prompt      : 인공지능은 교육에서
  Continuation: 개인별 학습 자료를 만드는 데 활용될 수 있습니다.
  ```
  - 활용: 이어쓰기, 대화, 요약문 생성, 코드 생성, 형식화된 응답
  - ⛔ 둘은 **분리된 세계가 아님** — 생성 모델도 내부 표현을 만들고, Encoder 모델도 Head를 붙이면 다양한 출력을 냄. 비교 대상은 **Objective가 직접 연습시키는 행동**
- **Downstream Task = 사전학습 모델을 실제 목적에 적용하는 구체적 문제**

  | Downstream Task | 입력 | 출력 | 자연스러운 후보 |
  | --- | --- | --- | --- |
  | 문의 intent 분류 | 문장 | 클래스 라벨 | Encoder-only + Classification Head |
  | 개인정보 토큰 탐지 | 문장 | 토큰별 라벨 | Encoder-only + Token Classification Head |
  | 문서 검색 | 질의·문서 | 임베딩·유사도 | Encoder 계열 임베딩 모델 |
  | 답변 초안 생성 | 질문·문맥 | 새 텍스트 | Decoder-only / Encoder-Decoder |
  | 번역 | 원문 | 번역문 | Encoder-Decoder |

  - ⭐ 모델 선택 순서: **문제의 입력 → 출력 → 필요한 문맥 방향**을 먼저 적고, 그다음 후보를 고름

- 📝 **이해도 점검**
  1. LM Objective를 한 문장으로? → 모델이 사전학습 중 **반복해서 풀도록 설계한 문제**
  2. 사전학습과 Fine-tuning의 차이는? → 대규모 비라벨에서 **일반 패턴** vs 소규모 목적별 데이터로 **문제 맞춤 추가 학습**
  3. 구조와 Objective를 따로 보는 이유는? → 같은 구조에 다른 Objective를 붙일 수 있고, 각각 **정보 흐름**과 **학습 문제**를 설명하므로
  4. 생성 태스크에 Decoder-only가 자연스러운 이유는? → **이전 토큰으로 다음 토큰을 예측**하는 학습이 텍스트를 순차 생성하는 과정과 그대로 이어지므로

**5장2강. BERT: Encoder-only와 Masked Language Modeling**
- **Bidirectional은 "문장을 거꾸로도 읽는다"가 아님**
  - ⭐ 각 토큰의 표현을 만들 때 **왼쪽 토큰과 오른쪽 토큰을 동시에 참고할 수 있다**는 뜻
  - Encoder의 Self-Attention에는 **causal mask가 없으므로** 모든 위치가 서로를 봄 (PAD mask만 적용)
- **Masked Language Modeling(MLM)** — 빈칸 채우기로 문맥을 학습
  ```
  원문 : 오늘 회의는 3시에 시작합니다
  입력 : 오늘 회의는 [MASK]에 시작합니다
  정답 : 3시   ← 선택된 위치의 "원래 token ID"
  ```
  - loss는 **선택된 위치에서만** 계산됨 (나머지 위치는 무시)
  - 관례적으로 전체 토큰의 약 **15%**를 학습 대상으로 고르고, 그중
    - **80%** → `[MASK]`로 치환
    - **10%** → 임의의 다른 토큰으로 치환
    - **10%** → 원래 토큰 그대로 유지
  - 💡 전부 `[MASK]`로 바꾸지 않는 이유 — 실제 Fine-tuning·추론 입력에는 `[MASK]`가 **없기 때문**. 학습 시점과 사용 시점의 입력 분포 차이(pretrain–finetune discrepancy)를 줄이려는 장치
  - ⛔ "10%를 그대로 두면 정답을 그냥 보여 주는 것 아닌가?" → 모델은 **어느 자리가 채점 대상인지 모르므로**, 바뀌지 않은 자리도 문맥으로 다시 확인해야 함
- **특수 토큰의 역할**

  | 토큰 | 역할 |
  | --- | --- |
  | `[CLS]` | 입력 시작 표시이자 **문장 대표 위치** (분류 Head가 주로 사용) |
  | `[SEP]` | 문장 끝 · **문장 쌍의 경계** |
  | `[MASK]` | MLM 학습·추론에서 **가려진 자리** 표시 |
  | `[PAD]` | 배치 안에서 **길이를 맞추기 위한 채움** (attention_mask로 제외) |

  - ⛔ 모델 계열마다 문자열이 다름 — BERT는 `[MASK]`, RoBERTa는 `<mask>`. **하드코딩 금지**
- **BERT 입력 임베딩 = Token + Position + Token Type**
  - `input_ids` : 토큰 ID
  - `attention_mask` : 실제 토큰 1 / PAD 0
  - `token_type_ids` : **문장 쌍**에서 첫 번째(0)·두 번째(1) 문장 구분 (단일 문장이면 전부 0)
- **Masked LM Head 출력**
  ```
  hidden state [B, L, D]  →  MLM Head  →  logits [B, L, V]
  ```
  - fill-mask에서 해석하는 위치는 **`input_ids == mask_token_id`인 자리**뿐

- 📝 **이해도 점검**
  1. BERT의 B가 뜻하는 것은? → 각 토큰이 **좌우 문맥을 함께 참고**하는 Bidirectional Context
  2. MLM의 정답은? → 학습 대상으로 선택된 위치의 **원래 token ID**
  3. `[MASK]`를 쓰는 목적은? → 정답을 그대로 보여 주지 않고 **주변 문맥으로 복원**하게 만들기 위해
  4. `token_type_ids`는 언제 쓰나? → **문장 쌍** 입력에서 첫 문장과 두 번째 문장을 구분할 때

**5장3강. fill-mask Pipeline 실습**
- **`pipeline("fill-mask")`가 묶어 주는 세 단계**
  ```
  ① Tokenization  ②  Masked LM forward  ③ 후처리(softmax → top-k → 문자열 복원)
  ```
  ```python
  from transformers import pipeline, AutoTokenizer

  MODEL_ID = "bert-base-uncased"
  tok = AutoTokenizer.from_pretrained(MODEL_ID)

  # ⭐ Mask 문자열을 하드코딩하지 말고 Tokenizer에서 확인
  print(tok.mask_token, tok.mask_token_id)      # 예: [MASK] 103

  fm = pipeline("fill-mask", model=MODEL_ID)
  out = fm(f"The capital of France is {tok.mask_token}.", top_k=5)
  # out[i] = {"sequence": ..., "score": ..., "token": ..., "token_str": ...}
  ```
- **결과 key가 담는 정보**

  | key | 의미 |
  | --- | --- |
  | `sequence` | 후보를 채워 **복원한 전체 문장** |
  | `score` | softmax 이후의 **모델 내부 후보 확률** |
  | `token` | 후보의 **token ID**(정수) |
  | `token_str` | 후보의 **문자열** |

  - ⛔ `top_k=10`은 "정답 10개"가 아니라 **점수 상위 10개 후보**를 뜻함
  - ⛔ `score`가 높다고 **사실**인 것은 아님 — score는 *모델 분포 안의 값*, 사실성은 *외부 세계와의 일치*
- **저수준 API로 직접 재현하기** — Pipeline이 감춘 계산을 눈으로 확인
  ```python
  import torch
  from transformers import AutoModelForMaskedLM

  model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).eval()
  enc = tok(f"The capital of France is {tok.mask_token}.", return_tensors="pt")

  with torch.no_grad():
      logits = model(**enc).logits                       # [B, L, V]

  pos = (enc["input_ids"] == tok.mask_token_id).nonzero()[0, 1]   # Mask 위치
  probs = logits[0, pos].softmax(dim=-1)                 # [V]
  top = probs.topk(5)
  print([tok.decode(i) for i in top.indices], top.values)
  ```
  - 직접 확인 가능한 것: 전체 **logits shape**, **Mask 위치**, vocabulary 분포, softmax·top-k 계산
- **결과를 정답처럼 쓰면 안 되는 이유**
  - 학습 데이터의 **언어·시기·도메인**이 대상 문장과 다를 수 있음
  - subword 때문에 후보가 **단어 조각**으로 나올 수 있음
  - 문맥 자체가 모호하면 여러 후보가 비슷한 score를 가짐

- 📝 **이해도 점검**
  1. Pipeline이 묶는 세 단계는? → **Tokenization → MLM forward → softmax·top-k·문자열 복원**
  2. Mask Token을 Tokenizer에서 확인하는 이유는? → 계열마다 **문자열과 ID가 다르기 때문**
  3. score와 사실성의 차이는? → score는 **모델 내부 후보 분포 값**, 사실성은 **외부 사실과의 일치 여부**
  4. 저수준 API로 더 볼 수 있는 것은? → logits shape, Mask 위치, vocabulary 분포, softmax·top-k **직접 계산**

**5장4강. GPT: Decoder-only와 Causal Language Modeling**
- **GPT = Generative(생성형) · Pre-trained(사전학습) · Transformer**
- **Decoder-only는 원래 Encoder-Decoder의 Decoder와 다름**
  - 별도 Encoder가 없고, Encoder memory를 참고하는 **Cross-Attention이 일반적으로 없음**
  - 남는 것은 **Causal Self-Attention + FFN** 블록의 반복
- **Causal Mask** — 미래를 가리는 장치
  - 각 위치가 **자기 자신과 왼쪽 토큰만** 보도록 attention 범위를 제한
  - ⭐ 이유는 **정보 누출 방지** — 다음 토큰을 맞추는 학습에서 정답이 입력에 미리 보이면 학습이 무의미해짐
  - 구현은 Softmax **이전에** 미래 위치 score를 `-inf`로 채우는 방식 (padding mask와 같은 원리)
- **Causal LM의 입력·정답 정렬** — 한 칸 밀기(shift)
  ```
  input :  나는  오늘  회의를  준비했다
  label :        오늘  회의를  준비했다  <eos>
            └──── 각 위치의 정답 = 한 칸 뒤 토큰 ────┘
  ```
  - 💡 그래서 **문장 하나로 여러 개의 학습 예제**가 동시에 생김
- **학습은 병렬, 생성은 순차**

  | 구분 | 특징 | 이유 |
  | --- | --- | --- |
  | 학습 | 모든 위치의 loss를 **한 번에** 계산 | 정답 시퀀스가 이미 있으므로 |
  | 생성 | 토큰을 **하나씩** 만들며 입력을 갱신 | 방금 만든 토큰이 다음 입력이므로 |

  - 💡 매 스텝 전체를 다시 계산하지 않도록 이전 K/V를 재사용하는 것이 **KV Cache** (그래서 K/V head 수가 추론 메모리에 직결됨)
- **다음 토큰 후보를 꺼내는 위치**
  ```python
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  tok = AutoTokenizer.from_pretrained("gpt2")
  model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

  enc = tok("인공지능은 교육에서", return_tensors="pt")
  with torch.no_grad():
      logits = model(**enc).logits          # [B, L, V]

  next_logits = logits[:, -1, :]            # ⭐ 마지막 "실제" 입력 위치 → [B, V]
  next_id = next_logits.argmax(dim=-1)

  out = model.generate(**enc, max_new_tokens=30, do_sample=False)
  print(tok.decode(out[0], skip_special_tokens=True))
  ```
  - ⛔ 배치에 오른쪽 PAD가 붙어 있으면 `-1`이 PAD 자리일 수 있음 → 배치 생성에서는 **left padding**을 쓰거나 `attention_mask`로 마지막 실제 위치를 찾아야 함
- **`generate()` 주요 인자**

  | 인자 | 역할 |
  | --- | --- |
  | `max_new_tokens` | **Prompt를 제외하고** 새로 생성할 최대 토큰 수 |
  | `do_sample` | `False`면 greedy, `True`면 확률적 샘플링 |
  | `temperature` | 분포를 평평하게(↑) 또는 뾰족하게(↓) 조절 |
  | `top_k` / `top_p` | 후보를 상위 k개 / 누적확률 p까지로 제한 |
  | `eos_token_id` | 종료 토큰이 나오면 생성 중단 |

  - 종료 조건은 **`max_new_tokens` 도달 · EOS 생성 · Stop Sequence** 등 여러 가지
  - ⛔ `max_length`는 **Prompt를 포함한** 전체 길이 기준이라 의미가 다름

- 📝 **이해도 점검**
  1. Causal Mask가 필요한 이유는? → 미래 정답 토큰을 미리 보는 **정보 누출**을 막기 위해
  2. Causal LM의 입력·정답 정렬은? → 각 입력 위치의 정답이 **한 칸 뒤 토큰**
  3. 학습과 생성의 병렬성 차이는? → 학습은 여러 위치 loss를 **병렬**, 생성은 직전 토큰이 필요해 **순차**
  4. next-token 후보는 어디서 꺼내나? → 현재 입력의 **마지막 실제 토큰 위치 logits `[B, V]`**
  5. `max_new_tokens`가 제한하는 것은? → **Prompt를 제외한** 새로 생성할 최대 토큰 수

**5장5강. BERT/GPT의 LM Head와 입력·출력 비교**
- **한눈에 비교**

  | 항목 | BERT 계열 | GPT 계열 |
  | --- | --- | --- |
  | 대표 구조 | Encoder-only | Decoder-only |
  | 문맥 방향 | 왼쪽 + 오른쪽 | 왼쪽 + 현재 |
  | 대표 Objective | Masked Language Modeling | Causal Language Modeling |
  | 사전학습 질문 | 가려진 토큰은 무엇인가? | 다음 토큰은 무엇인가? |
  | 대표 입력 | `[MASK]`가 포함된 문장 | Prompt |
  | 자연스러운 활용 | 분류 · 검색 · 토큰 이해 | 생성 · 대화 · 코드 |
  | 대표 클래스 | `AutoModelForMaskedLM` | `AutoModelForCausalLM` |

- **LM Head = 표현을 단어 점수로 바꾸는 마지막 층**
  ```
  hidden state [B, L, D]  →  LM Head  →  logits [B, L, V]
                             (마지막 축 D를 vocabulary 크기 V로 변환)
  ```
  - `B` Batch, `L` Sequence length, `D` Hidden dim, `V` Vocabulary size
  - 💡 많은 모델이 LM Head 가중치를 입력 embedding과 **공유(weight tying)** 해 파라미터를 절약함
- **같은 `[B, L, V]`라도 의미가 다르다**

  | 모델 | `logits[b, l, :]`의 의미 |
  | --- | --- |
  | BERT MLM | `l`번째 위치에 **원래 있었을** 토큰 후보 점수 |
  | GPT CLM | `l`번째 위치 **다음에 올** 토큰 후보 점수 |

  - ⭐ 그래서 **꺼내는 위치가 다름** — BERT는 `input_ids == mask_token_id`인 자리, GPT는 **마지막 실제 입력 위치**
  - ⛔ 두 모델의 **token ID 숫자를 직접 비교하면 안 됨** — vocabulary가 다르므로 같은 숫자가 같은 문자열을 뜻하지 않음. 반드시 **각 모델의 Tokenizer로 decode**
  - Pipeline 출력도 다름 — fill-mask는 후보별 `token_str`·`score`·`sequence` 목록, text-generation은 `generated_text` 중심 목록
- **필요한 출력 shape → 클래스 선택**

  | 해결할 문제 | 클래스 | 대표 출력 |
  | --- | --- | --- |
  | 토큰별 문맥 표현만 필요 | `AutoModel` | `[B, L, D]` hidden state |
  | Mask 채우기 | `AutoModelForMaskedLM` | `[B, L, V]` MLM logits |
  | 다음 토큰 생성 | `AutoModelForCausalLM` | `[B, L, V]` CLM logits |
  | 문장 분류 | `AutoModelForSequenceClassification` | `[B, C]` |
  | 토큰 분류 | `AutoModelForTokenClassification` | `[B, L, C]` |

- 📝 **이해도 점검**
  1. LM Head의 입출력 shape은? → `[B, L, D]` → **`[B, L, V]`**
  2. 두 모델의 logits shape이 비슷한 이유는? → 둘 다 **각 위치에서 vocabulary 후보를 평가**하므로
  3. 같은 shape인데 의미가 다른 이유는? → BERT는 **그 위치의 원래 토큰**, GPT는 **그 위치 다음 토큰**을 예측하므로
  4. `AutoModel`과 `AutoModelFor*`의 차이는? → Head 없이 **hidden state**만 vs 목적별 **Head를 붙인 logits**

**6장1강. Hugging Face Hub와 Model Card / License**
- **Hub = 모델 파일 + 문서 + 버전을 함께 관리하는 저장소**
  - **Model ID** = `조직/모델이름` → *어떤 저장소인가*
  - **revision** = branch · tag · **commit hash** → *그 저장소의 어떤 버전인가*
  - ⭐ 재현성이 중요하면 Model ID만이 아니라 **commit hash까지 기록**
  ```python
  model = AutoModel.from_pretrained("org/model-name", revision="a1b2c3d")
  ```
- **저장소에 보통 들어 있는 파일**

  | 파일 | 내용 |
  | --- | --- |
  | `config.json` | 아키텍처 · hidden size · label 매핑 등 모델 설정 |
  | `model.safetensors` | 가중치 (safetensors는 임의 코드 실행 없이 안전하게 로드) |
  | `tokenizer.json` · `vocab` · `tokenizer_config.json` | 토크나이저 규칙 · 어휘 |
  | `generation_config.json` | 기본 생성 파라미터 |

- **Model Card에서 최소한 확인할 항목**
  - `task` / `language` / `intended use` (와 out-of-scope use)
  - `training data` / `metrics` / `limitations & bias`
  - `license`
- **License 확인과 적합성 평가는 별개**
  - ⛔ 라이선스가 허용적이어도 **성능·안전성을 보장하지 않음** → 내부 평가는 별도 절차
  - 상업적 사용 가능 여부, 재배포·파생모델 조건, 사용 제한 조항을 각각 확인
- **Base Model vs Fine-tuned Model**

  | 구분 | 반환 | 용도 |
  | --- | --- | --- |
  | Base | 일반 문맥 표현 | 추가 학습 · 임베딩의 출발점 |
  | Fine-tuned | 특정 문제용 Head + 학습 결과 | 그 태스크에 바로 추론 |

- 📝 **이해도 점검**
  1. Model ID와 revision의 차이는? → **저장소**를 가리키는가, 그 저장소의 **특정 버전**을 가리키는가
  2. License가 허용적이면 평가를 생략해도 되나? → **아니오.** 라이선스는 성능·안전성과 무관
  3. Base와 Fine-tuned의 차이는? → 일반 표현만 vs **문제별 Head와 학습 결과 포함**

**6장2강. pipeline 기본 추론**
- **Pipeline이 묶는 네 단계**
  ```
  전처리(Tokenizer) → 모델 forward → 후처리 → Python 객체 반환
  ```
  - ⭐ **task name이 곧 계약** — 필요한 Head와 입력·출력 규칙이 task name으로 결정됨

  | task name | 필요한 Head | 대표 출력 |
  | --- | --- | --- |
  | `text-classification` | Sequence Classification Head | `label`, `score` |
  | `fill-mask` | Masked LM Head | `token_str`, `score`, `sequence` |
  | `text-generation` | Causal LM Head | `generated_text` |

  ```python
  from transformers import pipeline

  clf = pipeline("text-classification", model=CLS_MODEL_ID, device=0)   # device=0 → GPU
  gen = pipeline("text-generation", model="gpt2")

  clf(["배송이 너무 느려요", "품질이 훌륭합니다"])          # label / score 목록
  gen("인공지능은 교육에서", max_new_tokens=30)[0]["generated_text"]
  ```
- **Pipeline vs AutoClass — 목적이 다름**

  | Pipeline이 맞는 경우 | AutoClass가 맞는 경우 |
  | --- | --- |
  | 빠른 데모 · baseline 확인 | logits / hidden state 분석 |
  | 표준 후처리로 충분할 때 | custom batching |
  | 작은 입력 반복 | custom loss · 학습 · 디버깅 · 서비스 최적화 |

- 📝 **이해도 점검**
  1. Pipeline의 네 단계는? → **전처리 → forward → 후처리 → Python 객체 반환**
  2. task name이 중요한 이유는? → **필요한 Head와 입출력 규칙**을 결정하므로
  3. `text-generation` 결과의 대표 key는? → **`generated_text`**

**6장3강. AutoTokenizer와 AutoModel Base Output**
- **AutoClass는 어떻게 실제 클래스를 고르나**
  - Model ID 저장소의 **`config.json`(model_type · architectures)과 metadata**를 읽어 매핑된 구현 클래스를 선택
  - 💡 그래서 코드를 바꾸지 않고 **Model ID만 교체**해도 대부분 그대로 동작
- **Tokenizer 출력**
  ```python
  from transformers import AutoTokenizer, AutoModel
  import torch

  tok = AutoTokenizer.from_pretrained(MODEL_ID)
  enc = tok(["첫 문장", "두 번째는 조금 더 긴 문장입니다"],
            padding=True, truncation=True, return_tensors="pt")
  # input_ids [B, L] · attention_mask [B, L] · (모델에 따라) token_type_ids
  ```
- **Base AutoModel의 대표 출력 = `last_hidden_state` `[B, L, H]`**

  | 축 | 의미 |
  | --- | --- |
  | `B` | Batch (문장 수) |
  | `L` | Sequence length (토큰 수) |
  | `H` | Hidden size (토큰 표현 차원) |

  - ⛔ `pooler_output`은 **모델마다 없을 수 있고**, 문장 임베딩으로 항상 좋은 것도 아님 → pooling 방식은 태스크에 맞게 선택
- **Masked Mean Pooling** — PAD를 평균에서 빼야 함
  ```python
  model = AutoModel.from_pretrained(MODEL_ID).eval()
  with torch.no_grad():
      h = model(**enc).last_hidden_state              # [B, L, H]

  m = enc["attention_mask"].unsqueeze(-1)             # [B, L, 1]
  sent_emb = (h * m).sum(dim=1) / m.sum(dim=1)        # [B, H]
  ```
  - ⭐ PAD를 그대로 평균에 넣으면 **짧은 문장일수록 표현이 왜곡됨**

- 📝 **이해도 점검**
  1. AutoClass가 읽는 것은? → Model ID 저장소의 **config와 metadata**
  2. Base Model의 대표 출력은? → **`last_hidden_state`**, shape은 `[B, L, H]`
  3. masked mean pooling이 필요한 이유는? → **PAD 위치를 문장 평균에서 제외**하기 위해

**6장4강. Task-specific AutoModel과 logits 비교**
- **Task Head의 역할** — Base hidden representation을 **그 문제의 출력 형태로 변환**

  | 클래스 | 출력 shape | 의미 |
  | --- | --- | --- |
  | `AutoModelForSequenceClassification` | `[B, C]` | 문장별 클래스 점수 |
  | `AutoModelForTokenClassification` | `[B, L, C]` | 토큰별 클래스 점수 |
  | `AutoModelForMaskedLM` | `[B, L, V]` | 위치별 원래 토큰 후보 |
  | `AutoModelForCausalLM` | `[B, L, V]` | 위치별 다음 토큰 후보 |

  - `C`는 클래스 수, `V`는 vocabulary 크기 — **`[B, C]`는 문장 단위, `[B, L, V]`는 위치 단위**
  - GPT에서 다음 토큰 후보를 볼 때는 **마지막 실제 입력 토큰 위치**를 선택
- **label 문자열은 `model.config.id2label`에서 확인**
  ```python
  probs = logits.softmax(dim=-1)                       # [B, C]
  pred = probs.argmax(dim=-1)
  labels = [model.config.id2label[i.item()] for i in pred]
  ```
  - ⛔ `LABEL_0`, `LABEL_1`처럼 보이면 Model Card에서 **라벨 순서를 반드시 확인** (긍정/부정이 뒤집힐 수 있음)
  - ⛔ Head가 없는 체크포인트에 분류 클래스를 붙이면 **Head가 랜덤 초기화**된다는 경고가 뜸 → 그대로 추론하면 의미 없는 예측. Fine-tuning이 전제
- ⭐ 클래스 선택의 지름길: **"내가 필요한 출력 shape"을 먼저 적으면 클래스는 거의 자동으로 정해짐**

- 📝 **이해도 점검**
  1. Task Head의 역할은? → Base hidden representation을 **특정 문제의 출력으로 변환**
  2. `[B, C]`와 `[B, L, V]`의 차이는? → **문장별 클래스 점수** vs **위치별 vocabulary 점수**
  3. 분류 label 문자열은 어디서 확인하나? → **`model.config.id2label`**

**6장5강. Model·Tokenizer 저장/재로드와 추론 재현**
- **저장과 재로드는 한 쌍으로**
  ```python
  SAVE_DIR = "artifacts/my_model"
  model.save_pretrained(SAVE_DIR)
  tokenizer.save_pretrained(SAVE_DIR)      # ⭐ 반드시 같은 디렉터리에 함께

  model2 = AutoModelForCausalLM.from_pretrained(SAVE_DIR, local_files_only=True)
  tok2   = AutoTokenizer.from_pretrained(SAVE_DIR, local_files_only=True)
  ```
  - 모델만 저장하면 **같은 문자열이 같은 token ID로 변환된다는 보장이 사라짐** → 조용히 다른 결과가 나옴
  - `local_files_only=True` → **네트워크 없이 로컬 파일만** 사용 (캐시에 우연히 의존하는 상황을 차단)
- **재현 확인은 `assert_close`로**
  ```python
  torch.testing.assert_close(logits_a, logits_b, rtol=1e-4, atol=1e-5)
  ```
  - ⛔ 실수 logits에 `torch.equal()`은 부적절 — **하드웨어·dtype에 따른 미세한 부동소수점 오차**는 정상적으로 존재
  - 💡 최종 **prediction(argmax)** 일치까지 함께 확인하면 더 안전
- **재현성은 seed 하나가 아니라 "묶음"**
  - Model ID + **revision(commit hash)** + 파일 checksum
  - `transformers` · `torch` · CUDA/driver 버전
  - 전처리 설정(truncation · padding · max_length), **device / dtype**
  - `generation_config.json` 또는 실제 생성 파라미터, custom code와 `trust_remote_code` 사용 여부
  - rollback 대상 Artifact ID
- **선택 학습 · 폐쇄망(오프라인) 반입 체크리스트**
  - Private LLM 반입은 **폴더 복사로 끝나지 않음**
  1. 승인된 모델·Tokenizer **revision과 license**를 승인 기록에 남김
  2. weight · config · tokenizer · generation config · custom code가 모두 있는지 확인
  3. 파일별 **checksum manifest**를 만들어 반입 전후 무결성 비교
  4. 인터넷이 없는 환경에서 `local_files_only=True`로 재로드
  5. 대표 입력의 **logits/생성 결과 · dtype · 최대 VRAM · latency**를 baseline과 비교
  ```python
  from pathlib import Path
  from transformers import AutoModelForCausalLM, AutoTokenizer

  ARTIFACT_DIR = Path("artifacts/approved_causal_lm")   # 승인된 로컬 경로
  model = AutoModelForCausalLM.from_pretrained(ARTIFACT_DIR, local_files_only=True)
  tokenizer = AutoTokenizer.from_pretrained(ARTIFACT_DIR, local_files_only=True)
  ```
  - ⭐ 오프라인 테스트는 **새 캐시 경로 또는 격리된 환경**에서 — 그래야 "사실은 캐시가 있어서 됐던" 상황을 잡아냄

- 📝 **이해도 점검**
  1. 모델과 Tokenizer를 함께 저장하는 이유는? → 같은 문자열을 **같은 token ID·입력 형식**으로 재현하기 위해
  2. `local_files_only=True`의 역할은? → **네트워크 없이 로컬 파일만** 사용하도록 제한
  3. `assert_close`를 쓰는 이유는? → 하드웨어·dtype에 따른 **작은 부동소수점 오차를 허용**하기 위해
  4. 재현성 메타데이터에 남길 것은? → Model ID · revision, 라이브러리 버전, 전처리 설정, device/dtype, 입력·평가 조건

**오늘 배운 것 한눈에 정리**
- **구조(Architecture)와 Objective는 다른 축** — 구조는 *정보 흐름*, Objective는 *사전학습 중 반복해서 푸는 문제*
- BERT = **Encoder-only + Masked LM**("가려진 토큰은?"), GPT = **Decoder-only + Causal LM**("다음 토큰은?")
- Bidirectional은 거꾸로 읽는 것이 아니라 **각 토큰이 좌우 문맥을 동시에 참고**한다는 뜻
- MLM은 약 15%를 고르고 **80% `[MASK]` / 10% 랜덤 / 10% 유지** — 실제 입력에는 `[MASK]`가 없으므로 학습·사용 시점의 입력 차이를 줄이려는 장치
- Causal Mask는 **미래 정답 노출(정보 누출)을 막는 장치**이고, Causal LM의 정답은 **한 칸 뒤 토큰**
- ⭐ 학습은 여러 위치를 **병렬**로 계산하지만 생성은 직전 토큰이 필요해 **순차** — KV Cache가 여기서 의미를 가짐
- LM Head는 **`[B, L, D]` → `[B, L, V]`** 변환이며, shape이 같아도 **BERT는 "그 자리의 원래 토큰", GPT는 "그 자리 다음 토큰"** 으로 의미가 다름
- 그래서 꺼내는 위치도 다름 — **BERT는 Mask 위치, GPT는 마지막 실제 입력 위치**
- ⛔ 두 모델은 vocabulary가 다르므로 **token ID 직접 비교 금지** — 반드시 각 Tokenizer로 decode
- 모델 선택은 **필요한 출력 shape**부터: `[B,L,D]` → `AutoModel`, `[B,C]` → SequenceClassification, `[B,L,C]` → TokenClassification, `[B,L,V]` → MaskedLM / CausalLM
- Pipeline은 **전처리·forward·후처리·객체 반환**을 묶은 빠른 baseline, AutoClass는 **중간 Tensor·custom 처리·학습**용
- Model Card는 모델 선택의 출발점이며 **license 확인과 성능·안전성 평가는 별도 절차**
- 재현성은 seed 하나가 아니라 **revision · 환경 · 전처리 · dtype/device · 산출물 기록의 묶음**이고, 검증은 `local_files_only=True` 재로드 후 `assert_close`로

**데일리 퀴즈 정리**
1. fill-mask 태스크의 정의 → **문장 안 Mask Token 위치에 들어갈 후보를 문맥에 맞게 예측하는 태스크**
   - 검색 · 사실 검증 · 질의응답을 자동으로 수행해 주는 API가 **아님**
   - 모델 계열마다 Mask 문자열이 다르고(RoBERTa는 `<mask>`), **score는 사실성을 보장하지 않음**
2. BERT(MLM)와 GPT(CLM)의 차이 — **옳지 않은** 설명은 "같은 vocabulary를 쓰므로 token ID를 직접 비교해도 안전하다"
   - ⛔ 두 모델은 **서로 다른 Tokenizer·vocabulary**를 사용 → 반드시 각 모델의 Tokenizer로 **decode**해서 비교
   - 문맥 방향(좌우 vs 좌+현재)과 logits 해석 위치(Mask 자리 vs 마지막 실제 위치)도 서로 다름
3. `generate()`에서 Prompt 이후 새로 생성할 최대 토큰 수를 제한하는 인자 → **`max_new_tokens`**
   - ⛔ `max_tokens_length` 같은 이름이 아니며, **Prompt를 포함하는** `max_length`와도 구분해야 함
   - 종료 조건은 이 값 도달 · **EOS 토큰 생성** · Stop Sequence 등 여러 가지
4. Causal LM 학습에서 attention 범위를 왼쪽(이전) 토큰으로 제한하는 메커니즘 → **Causal Self-Attention(Causal Mask)**
   - 각 위치가 **오른쪽 미래 토큰을 보지 못하게 차단**해 정답이 입력에 미리 노출되는 것을 막음
   - GPT형 **Decoder-only 구조의 핵심 메커니즘**
5. 필요한 출력에 따른 AutoModel 클래스 선택 — **옳지 않은** 것은 "토큰별 hidden vector(`[B,L,D]`)만 필요할 때 → `AutoModelForCausalLM`"
   - Task Head 없이 `[B,L,D]`만 필요하면 **`AutoModel`(Base Model)**
   - `AutoModelForCausalLM`은 Causal LM Head가 붙어 **`[B,L,V]` vocabulary logits**를 반환하므로 목적이 다름

**한 줄 정리**: 사전학습 언어모델은 **구조(정보 흐름)와 Objective(반복해서 푸는 문제)를 따로 봐야** 하며 — BERT는 Encoder-only에 Masked LM을 붙여 좌우 문맥으로 "가려진 토큰"을 복원하면서 분류·검색에 쓸 **표현**을 학습하고, GPT는 Decoder-only에 Causal Mask를 걸어 왼쪽 문맥만으로 "다음 토큰"을 맞추며 **생성**을 학습한다. 두 모델의 LM Head는 똑같이 `[B,L,D] → [B,L,V]`를 만들지만 **BERT는 그 자리의 원래 토큰, GPT는 그 자리 다음 토큰**을 뜻하므로 꺼내는 위치(Mask 자리 vs 마지막 실제 위치)도 다르고 **vocabulary가 달라 token ID를 직접 비교해서는 안 된다**. 실무에서는 **필요한 출력 shape**(`[B,L,D]` · `[B,C]` · `[B,L,C]` · `[B,L,V]`)을 먼저 적어 AutoModel 클래스를 고르고, 빠른 baseline은 Pipeline으로 · 정밀 분석과 학습은 AutoClass로 나눠 쓰며, Model Card로 task·데이터·한계·license를 확인하되 **license 확인과 적합성 평가는 별개**로 두고, 마지막에는 model과 tokenizer를 **한 쌍으로 저장**해 `local_files_only=True`로 재로드한 뒤 `assert_close`로 logits를 비교하는 **revision·환경·dtype까지 포함한 재현성 묶음**을 남긴다.
