"""KOCOH v.2 사전학습 언어모델(PLM) 실험: Comment-only vs Context + Comment.

이 파일이 하는 일 (한 줄 요약)
    "baseline.py 와 같은 5-fold(cv_splits)에서 한국어 PLM 두 개를 (a) 고정 임베딩 + LogReg,
     (b) 전체 fine-tuning, (c) LoRA(PEFT) 세 방식으로 돌려, 맥락을 넣었을 때의 차이와 방식별 비용을 잰다."

이 파일은 PLM 학습의 유일한 진입점이다. 학습 방식은 fold 점수와 함께 test 행별 점수를 파일로 남기고,
지표·판정은 그 파일만 읽는 src/evaluate.py 가 계산한다 (학습 → 점수 저장 → 평가).

모델 (model/ 아래에 미리 받아 둠, 실행 중에는 내려받지 않음)
    - KcELECTRA-base-v2022 (beomi) : 네이버 뉴스 댓글로 사전학습 → 온라인 댓글 특화
    - klue-roberta-base   (klue)  : 뉴스·위키 등 일반 한국어로 사전학습

방법과 각각이 답하는 질문
    (a) frozen  : 모델은 고정, 마지막 층을 평균 풀링한 768차원 벡터 + LogReg(class_weight=balanced)
                  Q. 학습 없이 사전학습 표현만으로도 맥락이 도움이 되는가? (CPU로도 수 분)
    (b) finetune: 분류 헤드를 붙여 모델 전체(약 1.1~1.2억 파라미터)를 학습
                  Q. 모델이 맥락과 댓글의 관계를 직접 배우게 하면 맥락이 도움이 되는가?
    (c) lora    : 모델은 고정하고 어텐션 query·value 에 저랭크 행렬(r=8)만 붙여 학습 + 분류 헤드 학습
                  Q. 전체의 1% 안팎 파라미터만 학습해도 fine-tuning 만큼의 성능·맥락 이득이 나오는가?
                     그때 시간·메모리·저장 크기는 얼마나 줄어드는가?

입력 조건 (앞의 둘이 기본, 뒤의 둘은 --conditions 로 추가)
    Comment-only   : [CLS] Comment [SEP]
    Context+Comment: [CLS] Context [SEP] Comment [SEP]   (문장쌍 입력, 같이 읽기 = cross-encoder)
    Context-only   : [CLS] Context [SEP]                 (댓글 없이 글만 보고 맞힐 수 있는 몫 확인용)
    Bi-encoder     : 글과 댓글을 같은 인코더로 따로 읽고 [CLS] 두 개를 이어 붙여 판단 (따로 읽기, fine-tuning 전용)
    max_length=128 — 문장쌍 3,000개 중 128 토큰을 넘는 것은 5~8개뿐 (나머지는 잘리지 않음)

자원 규칙 (Colab 무료 GPU, 학습 1회 30분 이내)
    - 하이퍼파라미터는 고정값만 쓰고 test fold를 보고 고르지 않는다 (검증 셋이 없으므로 조기 종료도 없음).
      공통: epoch 3, batch 32, warmup 10%, weight decay 0.01, 클래스 가중 CE
      finetune: lr 3e-5 / lora: lr 5e-4, r=8, alpha=16, dropout 0.1 (LoRA 는 학습 파라미터가 적어 lr 을 크게 쓰는 것이 일반적)
    - 학습 시간(train_sec)은 기기가 잠든 시간을 빼는 monotonic 시계로 재고, 실제 경과 시간과의 차이를
      sleep_sec 으로 따로 기록한다. 30분을 넘으면 경고를 출력한다.
    - 메모리(peak_mem_mb): 학습 중 할당된 GPU 메모리의 최대치 (CUDA: max_memory_allocated,
      MPS: 매 step 의 current_allocated_memory 중 최대 — 근사값).
    - CUDA(Colab)에서는 fp16 autocast, Apple MPS·CPU에서는 fp32.

결과 파일 (results/)
    <이름>.csv        : (모델, 방식, 조건, seed, fold)별 점수와 비용. 기본 이름은 plm_folds
    <이름>_preds.csv  : 학습 방식(finetune·lora)의 test 행별 점수. 뒤에 이어 붙이므로 끊겨도 앞선 결과가 남는다.
        열: model, method, condition, seed, fold, Index, y, score, variant
        score   = 혐오 logit − 비혐오 logit. 0 보다 크면 혐오로 예측한다 (argmax 와 같음, 확률로는 sigmoid(score)).
        variant = 붙인 글의 종류. 원래 입력은 "own", Context 를 바꿔 끼운 입력은 그 이름 (run_trained 의 extras).
    <이름>_inner_preds.csv : --inner 로 만든 train 내부 validation 행별 점수 (판정 임계값 선택용, test 는 쓰지 않음).
        fold 마다 train 을 게시글 묶음 단위 4조각으로 나눠(baseline.py 의 OOF 조정과 같은 분할) 3조각으로 학습하고
        남은 1조각을 예측한다. 열: model, method, condition, seed, fold, inner_fold, Index, y, score

터미널에서 실행
    python src/plm.py                                    # 전부: 2모델 × 3방식 × 2조건 × 5fold = 60개
    python src/plm.py --methods frozen                   # 고정 임베딩만 (빠름)
    python src/plm.py --models klue-roberta-base --folds 0 --methods lora   # 시간 측정용

    # 가설 1: 읽는 방식 비교. 끊겨도 같은 명령을 다시 실행하면 끝난 조합은 건너뛴다 (--resume)
    python src/plm.py --models KcELECTRA-base-v2022 --methods finetune --out results/plm_h1.csv --resume --seeds 42 43 44 --conditions Comment-only Context+Comment Context-only Bi-encoder
    python src/evaluate.py --preds results/plm_h1_preds.csv

    # 임계값 선택용 내부 validation 점수 (test 모델은 다시 학습하지 않는다). 끊기면 같은 명령을 다시 실행한다
    python src/plm.py --models KcELECTRA-base-v2022 --methods finetune --out results/plm_h1.csv --inner --conditions Comment-only Context+Comment Context-only Bi-encoder
"""
import argparse
import atexit
import math
import os
import time
from datetime import date
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"  # model/ 에 받아 둔 파일만 쓴다
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import sklearn
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from preprocess import ROOT, SEED, SPLIT_CSV, cv_splits, cv_splits_inner, get_xy

hf_logging.set_verbosity_error()

MODELS = {
    "KcELECTRA-base-v2022": ROOT / "model" / "KcELECTRA-base-v2022",
    "klue-roberta-base": ROOT / "model" / "klue-roberta-base",
}
METHODS = ["frozen", "finetune", "lora"]
# 조건 이름 → 읽는 방식
CONDITIONS = {"Comment-only": "comment", "Context+Comment": "cross", "Context-only": "context", "Bi-encoder": "bi"}
DEFAULT_CONDITIONS = ["Comment-only", "Context+Comment"]
METRICS = ["accuracy", "macro_f1", "recall_hate"]
OUT_CSV = ROOT / "results" / "plm_folds.csv"
PRED_COLS = ["model", "method", "condition", "seed", "fold", "Index", "y", "score", "variant"]
INNER_COLS = ["model", "method", "condition", "seed", "fold", "inner_fold", "Index", "y", "score"]
ARRAY_KEYS = ("score", "extra_score", "extra_proba")  # info 에 담기는 행별 배열 (fold 점수 행에는 넣지 않는다)

MAX_LEN = 128
EPOCHS, BATCH, WARMUP, WEIGHT_DECAY = 3, 32, 0.1, 0.01
LR = {"finetune": 3e-5, "lora": 5e-4}
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])
TIME_LIMIT_SEC = 30 * 60  # 학습 1회 상한

rows = []  # fold별 점수
RUN = {"expected": None}  # atexit 실험로그용


# ───────────────────────────── 0. 공통 ─────────────────────────────
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU·CUDA·MPS 모두 고정 (MPS는 연산 자체가 완전히 결정적이지 않을 수 있음)


class Timer:
    """학습 시간 측정. monotonic 은 기기가 잠든 동안 멈추므로 실제 계산 시간만 잰다.
    벽시계(time.time)와의 차이 = 잠든 시간 → sleep_sec 으로 기록해 오염 여부를 확인할 수 있게 한다."""

    def __enter__(self):
        self.m0, self.w0 = time.monotonic(), time.time()
        return self

    def __exit__(self, *exc):
        self.sec = time.monotonic() - self.m0
        self.sleep_sec = max(0.0, (time.time() - self.w0) - self.sec)


def gpu_mem_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return float("nan")


def as_mode(mode):
    """읽는 방식 이름으로 맞춘다. 예전 호출의 use_context(True/False)도 받는다."""
    if isinstance(mode, bool):
        return "cross" if mode else "comment"
    return mode


def texts(df, mode):
    """get_xy 로 입력을 꺼내 토크나이저에 넣을 형태로 돌려준다: (첫 문장, 둘째 문장 또는 None, 정답)."""
    X, y = get_xy(df, use_context=mode != "comment")
    if mode == "comment":
        return X["Comment"].tolist(), None, y
    if mode == "context":
        return X["Context"].tolist(), None, y
    return X["Context"].tolist(), X["Comment"].tolist(), y  # cross 는 문장쌍으로, bi 는 각각 토큰화한다


def encode(tok, first, second, config):
    """문장(또는 문장쌍)을 토큰화한다. 패딩은 배치마다 따로 하므로 여기서는 하지 않는다.

    klue-roberta-base 는 문장 구분 임베딩(type_vocab_size)이 1개뿐이라, 토크나이저가 둘째 문장에 붙이는
    token_type_ids=1 을 넣으면 범위 밖 오류가 난다 → 그런 모델에서는 token_type_ids 를 빼고 [SEP] 로만 구분한다.
    """
    enc = tok(first, second, truncation=True, max_length=MAX_LEN)
    if getattr(config, "type_vocab_size", 2) < 2:
        enc.pop("token_type_ids", None)
    return enc


def encode_inputs(tok, df, mode, config):
    """df 를 읽는 방식에 맞게 토큰화한다: (enc, 정답). 따로 읽기는 enc 가 (맥락, 댓글) 튜플이다."""
    first, second, y = texts(df, mode)
    if mode == "bi":
        return (encode(tok, first, None, config), encode(tok, second, None, config)), y
    return encode(tok, first, second, config), y


def pad(enc, b, tok, device):
    """인덱스 b 의 행을 그 안에서 가장 긴 길이에 맞춰 패딩한다."""
    feats = [{k: enc[k][i] for k in enc.keys()} for i in b]
    return {k: v.to(device) for k, v in tok.pad(feats, return_tensors="pt").items()}


def batches(enc, idx, tok, device, labels=None):
    """인덱스 순서대로 BATCH 개씩 잘라, 배치 안에서 가장 긴 길이에 맞춰 패딩한 텐서를 낸다.
    따로 읽기는 enc 가 (맥락, 댓글) 튜플이고, 둘을 각각 패딩한다."""
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        if isinstance(enc, tuple):
            out = {"context": pad(enc[0], b, tok, device), "comment": pad(enc[1], b, tok, device)}
        else:
            out = pad(enc, b, tok, device)
        if labels is not None:
            out["labels"] = torch.as_tensor(labels[b], device=device)
        yield out


def pred_csv():
    """행별 점수 파일 경로. fold 점수 파일 옆에 둔다: plm_h1.csv → plm_h1_preds.csv"""
    return OUT_CSV.with_name(OUT_CSV.stem + "_preds.csv")


def check_pred_format():
    """이어 붙일 행별 점수 파일의 열이 현재 형식인지 확인한다. 학습을 시작하기 전에 불러 시간을 버리지 않게 한다."""
    path = pred_csv()
    if path.exists() and pd.read_csv(path, nrows=0).columns.tolist() != PRED_COLS:
        raise ValueError(f"{path.name} 의 열이 현재 형식({', '.join(PRED_COLS)})과 다릅니다. --out 으로 새 이름을 쓰세요.")


def save_preds(model, method, cond, fold, seed, index, y_true, s, variant="own"):
    """행별 점수를 쌓는다. 파일을 덮어쓰지 않고 뒤에 붙여서 끊겨도 앞선 결과가 남는다."""
    path = pred_csv()
    check_pred_format()
    pd.DataFrame({"model": model, "method": method, "condition": cond, "seed": seed, "fold": fold,
                  "Index": np.asarray(index), "y": y_true, "score": np.round(s, 5), "variant": variant})[PRED_COLS].to_csv(
        path, mode="a", header=not path.exists(), index=False)


def score(model, method, cond, fold, y_true, y_pred, device, info, seed=SEED):
    """fold 하나의 점수와 비용 정보(info: train_sec, sleep_sec, trainable_params, ...)를 한 행으로 저장한다."""
    rows.append({
        "model": model,
        "method": method,
        "condition": cond,
        "seed": seed,
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_hate": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        **{k: v for k, v in info.items() if k not in ARRAY_KEYS},
        "device": device.type,
    })
    pd.DataFrame(rows).round(6).to_csv(OUT_CSV, index=False)  # 중간에 끊겨도 그때까지 결과가 남도록 매번 저장
    r = rows[-1]
    flag = "  ⚠ 30분 초과" if r["train_sec"] > TIME_LIMIT_SEC else ""
    flag += f"  (잠자기 {r['sleep_sec']:.0f}s 제외)" if r["sleep_sec"] > 5 else ""
    print(f"  [{model} | {method} | {cond} | seed {seed} | fold {fold}] macro-F1={r['macro_f1']:.3f} "
          f"recall_hate={r['recall_hate']:.3f} acc={r['accuracy']:.3f} ({r['train_sec']:.0f}s){flag}")


# ───────────────────────────── (a) 고정 임베딩 + LogReg ─────────────────────────────
@torch.no_grad()
def embed_all(model_dir, df, mode, device):
    """전체 행의 평균 풀링 벡터를 한 번에 구한다. 모델이 고정이라 fold와 무관 → 학습 정보 누설 없음."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc_model = AutoModel.from_pretrained(model_dir).to(device).eval()
    first, second, _ = texts(df, mode)
    enc = encode(tok, first, second, enc_model.config)
    out = []
    for b in batches(enc, np.arange(len(first)), tok, device):
        h = enc_model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())  # 패딩 제외 평균
    total = sum(p.numel() for p in enc_model.parameters())
    return np.vstack(out), total


def run_frozen(name, folds, device, conds=DEFAULT_CONDITIONS):
    """조건마다 전체 df 임베딩을 한 번 구해 두고, fold마다 그 fold의 train 행으로만 LogReg를 학습한다."""
    full = pd.concat([folds[0][1], folds[0][2]])  # 어느 fold든 train+test = 전체 3000행 (원문 텍스트는 전처리와 무관)
    key = full["Index"].to_numpy()
    pos = pd.Series(np.arange(len(key)), index=key)
    for cond in conds:
        with Timer() as t_emb:
            E, total = embed_all(MODELS[name], full, CONDITIONS[cond], device)
        for fold, train, test in folds:
            tr_i, te_i = pos[train["Index"]].to_numpy(), pos[test["Index"]].to_numpy()
            with Timer() as t:
                clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)
                clf.fit(E[tr_i], train["Hate speech"].to_numpy())
            info = {
                # 학습 시간 = 임베딩 계산(전체를 fold 수로 나눈 몫) + LogReg 학습
                "train_sec": round(t_emb.sec / len(folds) + t.sec, 1),
                "sleep_sec": round(t_emb.sleep_sec / len(folds) + t.sleep_sec, 1),
                "trainable_params": int(clf.coef_.size + clf.intercept_.size),
                "total_params": int(total),
                "peak_mem_mb": float("nan"),
                "train_loss": float("nan"),
            }
            score(name, "frozen", cond, fold, test["Hate speech"].to_numpy(), clf.predict(E[te_i]), device, info)


# ───────────────────────────── (b) 전체 fine-tuning / (c) LoRA ─────────────────────────────
class BiEncoder(torch.nn.Module):
    """따로 읽기: 같은 인코더로 맥락과 댓글을 각각 읽고 [CLS] 두 개를 이어 붙여 판단한다.

    두 문장의 토큰은 서로를 보지 못하므로 self-attention 으로 섞이지 않는다. 같이 읽기와 다른 점이 이것뿐이도록
    첫 토큰을 쓰고 헤드도 ElectraClassificationHead 와 같은 구조(dropout-은닉층-GELU-dropout-출력)로 둔다.
    은닉층이 있어서 맥락과 댓글을 '판단 단계'에서 조합할 수는 있다. 선형 층 하나면 조합 자체가 불가능해
    실패해도 헤드 탓인지 읽는 방식 탓인지 구분할 수 없다.
    """

    def __init__(self, model_dir):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_dir)
        self.config = self.enc.config
        d, p = self.config.hidden_size, self.config.hidden_dropout_prob
        self.head = torch.nn.Sequential(torch.nn.Dropout(p), torch.nn.Linear(2 * d, d), torch.nn.GELU(),
                                        torch.nn.Dropout(p), torch.nn.Linear(d, 2))

    def forward(self, context, comment):
        u = self.enc(**context).last_hidden_state[:, 0]
        v = self.enc(**comment).last_hidden_state[:, 0]
        return SimpleNamespace(logits=self.head(torch.cat([u, v], -1)))


def predict_logits(model, enc, n, tok, device, use_amp):
    """n 개 행의 logit (n × 2). 행 순서는 입력 그대로다."""
    out = []
    with torch.no_grad():
        for b in batches(enc, np.arange(n), tok, device):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out.append(model(**b).logits.float().cpu())
    return torch.cat(out).numpy()


def finetune_one(model_dir, train, test, mode, device, method="finetune", seed=SEED, extra_tests=None):
    """fold 하나를 학습하고 (정답, test 예측, 비용 정보) 를 돌려준다. 하이퍼파라미터는 고정값만 쓴다.

    mode             : 읽는 방식 ("comment" / "cross" / "context" / "bi"). 예전 호출의 True/False 는 cross/comment 로 읽는다.
    method="finetune": 모델 전체 학습
    method="lora"    : 원래 가중치는 고정, query·value 의 LoRA 행렬과 분류 헤드만 학습
    extra_tests      : {이름: df} — 학습이 끝난 같은 모델로 추가 입력(예: Context 를 바꿔 끼운 test)을 예측한다.
                       학습에는 쓰지 않는다.

    info 에는 비용 정보와 함께 행별 결과가 담긴다 (행 순서는 입력 df 그대로).
        info["score"]                    : test 행별 점수 = 혐오 logit − 비혐오 logit
        info["extra_score"][이름]         : extra_tests 의 행별 점수
        info["extra_proba"][이름]         : 같은 값을 혐오 확률로 바꾼 것 (= sigmoid(score))
    """
    mode = as_mode(mode)
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_dir)
    if mode == "bi":
        model = BiEncoder(model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    if method == "lora":
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA))  # 분류 헤드는 peft 가 학습 대상으로 둔다
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    enc_tr, y_tr = encode_inputs(tok, train, mode, model.config)
    enc_te, y_te = encode_inputs(tok, test, mode, model.config)

    # 클래스 불균형: baseline 의 class_weight="balanced" 와 같은 가중치 (train 비율로만 계산)
    counts = np.bincount(y_tr, minlength=2)
    weight = torch.tensor(len(y_tr) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    steps = EPOCHS * math.ceil(len(y_tr) / BATCH)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR[method], weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    peak = 0.0
    with Timer() as t:
        model.train()
        for _ in range(EPOCHS):
            losses = []
            for b in batches(enc_tr, rng.permutation(len(y_tr)), tok, device, labels=y_tr):
                labels = b.pop("labels")
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = loss_fn(model(**b).logits.float(), labels)
                peak = max(peak, gpu_mem_mb(device))  # 순전파 직후 (활성값이 남아 있을 때)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                peak = max(peak, gpu_mem_mb(device))  # 갱신 직후 (기울기·옵티마이저 상태가 있을 때)
                losses.append(loss.item())
        if device.type == "mps":
            torch.mps.synchronize()

    model.eval()
    logits = predict_logits(model, enc_te, len(y_te), tok, device, use_amp)
    extra_score = {}
    for key, df in (extra_tests or {}).items():
        enc_x, y_x = encode_inputs(tok, df, mode, model.config)
        lx = predict_logits(model, enc_x, len(y_x), tok, device, use_amp)
        extra_score[key] = lx[:, 1] - lx[:, 0]
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    info = {
        "train_sec": round(t.sec, 1),
        "sleep_sec": round(t.sleep_sec, 1),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "peak_mem_mb": round(peak, 1),
        "train_loss": round(float(np.mean(losses)), 4),  # 마지막 epoch 평균 (학습이 수렴했는지 확인용, test 와 무관)
        "score": logits[:, 1] - logits[:, 0],
    }
    if extra_tests:
        info["extra_score"] = extra_score
        info["extra_proba"] = {k: 1 / (1 + np.exp(-v)) for k, v in extra_score.items()}
    return y_te, logits.argmax(-1), info


def run_trained(name, folds, device, method, conds=DEFAULT_CONDITIONS, seeds=(SEED,), done=(), extras=None):
    """seed 하나를 끝낸 뒤 다음 seed 로 넘어간다. done 에 있는 조합은 건너뛴다 (--resume).
    seed 는 모델 초기값과 배치 순서만 바꾼다. 데이터 분할은 언제나 SEED 로 고정이다.

    extras: (test, fold) → {이름: df} 를 돌려주는 함수. 같은 모델로 그 입력들도 예측해
            행별 점수 파일에 variant=이름 으로 함께 저장한다 (예: Context 를 바꿔 끼운 test).
    """
    for seed in seeds:
        for fold, train, test in folds:
            for cond in conds:
                if (name, method, cond, seed, fold) in done:
                    continue
                extra_tests = extras(test, fold) if extras else None
                y_te, pred, info = finetune_one(MODELS[name], train, test, CONDITIONS[cond], device,
                                                method=method, seed=seed, extra_tests=extra_tests)
                index = test["Index"].to_numpy()
                save_preds(name, method, cond, fold, seed, index, y_te, info["score"])
                for key, s in info.get("extra_score", {}).items():
                    save_preds(name, method, cond, fold, seed, index, y_te, s, variant=key)
                score(name, method, cond, fold, y_te, pred, device, info, seed=seed)


def check_split(folds):
    """test fold 가 팀의 기준 분할 기록(results/split_assignment.csv)과 같은지 학습 전에 확인하고, 다르면 멈춘다.

    scikit-learn 버전에 따라 StratifiedGroupKFold(shuffle=True)가 다른 분할을 만들 수 있다.
    분할이 다르면 기준선·다른 가설과 fold 가 맞지 않아 결과를 함께 쓸 수 없다 (README 의 '알아 둘 것').
    """
    if not SPLIT_CSV.exists():
        print(f"알림: {SPLIT_CSV.name} 가 없어 분할을 기록과 대조하지 못했습니다.")
        return
    rec = pd.read_csv(SPLIT_CSV)
    rec = rec[rec["role"] == "test"].set_index("Index")["outer_fold"]
    for fold, _, test, *_ in folds:
        if set(test["Index"]) != set(rec.index[rec == fold]):
            raise SystemExit(f"중단: fold {fold} 의 test 행이 기준 분할 기록({SPLIT_CSV.name})과 다릅니다. "
                             f"이 환경(scikit-learn {sklearn.__version__})이 기록과 다른 분할을 만들었습니다. "
                             f"requirements.txt 의 scikit-learn 버전을 설치한 뒤 다시 실행하세요.")
    print(f"분할 확인: test fold 가 {SPLIT_CSV.name} 기록과 같습니다.")


def inner_csv():
    """내부 validation 행별 점수 파일 경로: plm_h1.csv → plm_h1_inner_preds.csv"""
    return OUT_CSV.with_name(OUT_CSV.stem + "_inner_preds.csv")


def run_inner(name, folds, device, method, conds=DEFAULT_CONDITIONS, seeds=(SEED,)):
    """판정 임계값 선택용: fold 마다 train 안에서만 학습·예측한 점수(OOF)를 쌓는다. test 는 쓰지 않는다.

    folds: cv_splits_inner() 가 주는 (fold, train, test, inner). inner = [(inner-train 위치, validation 위치), ...]
    조건·하이퍼파라미터는 test 모델과 같고 학습 행만 train 의 3/4 이다. 이미 파일에 있는 조합은 건너뛴다.
    임계값은 src/evaluate.py 가 이 점수에서 고른다 (고른 뒤 적용하는 test 점수는 train 전체로 학습한 모델의 것).
    """
    path = inner_csv()
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        if old.columns.tolist() != INNER_COLS:
            raise ValueError(f"{path.name} 의 열이 현재 형식({', '.join(INNER_COLS)})과 다릅니다. --out 으로 새 이름을 쓰세요.")
        done = set(old[INNER_COLS[:6]].drop_duplicates().itertuples(index=False, name=None))
        print(f"이어서 실행: {path.name} 에 내부 학습 {len(done)}개가 있습니다.")
    n = 0
    for seed in seeds:
        for fold, train, _, inner in folds:
            for cond in conds:
                for k, (itr, iva) in enumerate(inner):
                    if (name, method, cond, seed, fold, k) in done:
                        continue
                    val = train.iloc[iva]
                    y_va, pred, info = finetune_one(MODELS[name], train.iloc[itr], val, CONDITIONS[cond], device,
                                                    method=method, seed=seed)
                    pd.DataFrame({"model": name, "method": method, "condition": cond, "seed": seed, "fold": fold,
                                  "inner_fold": k, "Index": val["Index"].to_numpy(), "y": y_va,
                                  "score": np.round(info["score"], 5)})[INNER_COLS].to_csv(
                        path, mode="a", header=not path.exists(), index=False)
                    n += 1
                    flag = "  ⚠ 30분 초과" if info["train_sec"] > TIME_LIMIT_SEC else ""
                    print(f"  [{name} | {method} | {cond} | seed {seed} | fold {fold} | 내부 {k}] validation {len(val)}행 "
                          f"acc={accuracy_score(y_va, pred):.3f} ({info['train_sec']:.0f}s){flag}")
    return n


# ───────────────────────────── 실험로그 (atexit) ─────────────────────────────
def mean_std(s):
    """평균 ± 표준편차(표본, ddof=1) 문자열."""
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"


def log_row():
    """예정된 조합이 모두 끝났을 때만 실험로그 한 행을 출력한다."""
    if RUN["expected"] is None or len(rows) != RUN["expected"]:
        print(f"\n결과 미완 — 로그 생략 (모인 결과 {len(rows)} / {RUN['expected']})")
        return
    df = pd.DataFrame(rows)
    f1 = df.groupby(["model", "method", "condition", "fold"], sort=False)["macro_f1"].mean()  # seed 평균을 먼저
    f1 = f1.groupby(level=["model", "method", "condition"], sort=False)
    parts = [f"macro_f1[{m} / {me} / {c}]={mean_std(s)}" for (m, me, c), s in f1]
    parts.append(f"train_sec_max={df['train_sec'].max():.0f}")
    config = (f"split=post_group 5-fold StratifiedGroupKFold, seed={SEED}, max_len={MAX_LEN}, "
              f"epoch={EPOCHS}, batch={BATCH}, lr_finetune={LR['finetune']}, lr_lora={LR['lora']}, "
              f"lora_r={LORA['r']}, device={df['device'].iloc[0]}")
    print("\n[실험로그]")
    print("| 실험 이름 | 설정 | 지표 | 결과 파일 | 날짜 |")
    print("|---|---|---|---|---|")
    print(f"| plm | {config} | {', '.join(parts)} | {OUT_CSV.relative_to(ROOT)} | {date.today().isoformat()} |")
    print("\nNotion 기록은 제안만 함 (실행하지 않음): src.notion_log.log_experiment(\"plm\", ...)")


# ───────────────────────────── 터미널에서 직접 실행할 때 ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default=None, help="결과 CSV 경로 (기본 results/plm_folds.csv)")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED], help="모델 초기값 seed (분할은 고정)")
    parser.add_argument("--resume", action="store_true", help="결과 파일을 이어 쓰고 끝난 조합은 건너뜀")
    parser.add_argument("--inner", action="store_true",
                        help="임계값 선택용 train 내부 validation 점수만 만든다 (test 모델은 학습하지 않음, 이어 돌리기 자동)")
    args = parser.parse_args()
    if args.out:
        OUT_CSV = ROOT / args.out
    if args.inner:
        if "frozen" in args.methods or ("Bi-encoder" in args.conditions and args.methods != ["finetune"]):
            parser.error("--inner 는 학습 방식만 (frozen 제외), Bi-encoder 는 finetune 으로만 돌린다")
        device = pick_device()
        print(f"seed = {SEED}, device = {device}")
        for name in args.models:
            if not (MODELS[name] / "config.json").exists():
                raise FileNotFoundError(f"{MODELS[name]} 가 없습니다. model/ 에 먼저 받아 두세요.")
        folds = [f for f in cv_splits_inner(save=False) if f[0] in args.folds]
        check_split(folds)
        n = sum(run_inner(name, folds, device, method, args.conditions, args.seeds)
                for name in args.models for method in args.methods)
        print(f"\n저장: {inner_csv().relative_to(ROOT)} (이번에 내부 학습 {n}개 추가)")
        raise SystemExit(0)
    if "frozen" in args.methods and (args.resume or set(args.conditions) != set(DEFAULT_CONDITIONS) or args.seeds != [SEED]):
        parser.error("frozen 은 기본 조건·기본 seed 로만, --resume 없이 돌린다")
    if "Bi-encoder" in args.conditions and args.methods != ["finetune"]:
        parser.error("Bi-encoder 는 finetune 으로만 돌린다")

    atexit.register(log_row)
    set_seed()
    device = pick_device()
    print(f"seed = {SEED}, device = {device}")
    for name in args.models:
        if not (MODELS[name] / "config.json").exists():
            raise FileNotFoundError(f"{MODELS[name]} 가 없습니다. model/ 에 먼저 받아 두세요.")

    done = set()
    if args.resume and OUT_CSV.exists():
        check_pred_format()  # 학습 전에 확인
        rows.extend(pd.read_csv(OUT_CSV).to_dict("records"))
        done = {(r["model"], r["method"], r["condition"], r.get("seed", SEED), r["fold"]) for r in rows}
        print(f"이어서 실행: {OUT_CSV.relative_to(ROOT)} 에 {len(rows)}개 조합이 있습니다.")
    elif args.resume and pred_csv().exists():
        parser.error(f"{pred_csv().name} 는 있는데 {OUT_CSV.name} 가 없습니다. 이어 돌릴 기록이 없어 멈춥니다 "
                     f"(행별 점수를 지우지 않도록). 새로 돌리려면 --out 에 다른 이름을 쓰세요.")
    elif OUT_CSV.exists() or pred_csv().exists():
        print(f"알림: {OUT_CSV.relative_to(ROOT)} 이(가) 이미 있어 덮어씁니다.")
        pred_csv().unlink(missing_ok=True)
    folds = [(f, tr, te) for f, tr, te, _ in cv_splits() if f in args.folds]
    check_split(folds)
    planned = {(n, me, c, s, f) for n in args.models for me in args.methods for c in args.conditions
               for s in args.seeds for f, _, _ in folds}
    RUN["expected"] = len(done | planned)

    for name in args.models:
        for method in args.methods:
            print(f"\n=== {name} — {method} ===")
            if method == "frozen":
                run_frozen(name, folds, device, args.conditions)
            else:
                run_trained(name, folds, device, method, args.conditions, args.seeds, done)
    print(f"\n저장: {OUT_CSV.relative_to(ROOT)} ({len(rows)}행)")
    if pred_csv().exists():
        print(f"저장: {pred_csv().relative_to(ROOT)} (test 행별 점수)")
