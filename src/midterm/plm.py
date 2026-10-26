"""KOCOH v.2 사전학습 언어모델(PLM) 실험: Comment-only vs Context + Comment.

이 파일이 하는 일 (한 줄 요약)
    "baseline.py 와 같은 5-fold(cv_splits)에서 한국어 PLM 두 개를 (a) 고정 임베딩 + LogReg,
     (b) 전체 fine-tuning, (c) LoRA(PEFT) 세 방식으로 돌려, 맥락을 넣었을 때의 차이와 방식별 비용을 잰다."

모델 (models/ 아래에 미리 받아 둠, 실행 중에는 내려받지 않음)
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

입력 조건
    Comment-only   : [CLS] Comment [SEP]
    Context+Comment: [CLS] Context [SEP] Comment [SEP]   (문장쌍 입력)
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

터미널에서 실행
    python -m src.midterm.plm                                    # 전부: 2모델 × 3방식 × 2조건 × 5fold = 60개
    python -m src.midterm.plm --methods frozen                   # 고정 임베딩만 (빠름)
    python -m src.midterm.plm --models klue-roberta-base --folds 0 --methods lora   # 시간 측정용
"""
import argparse
import atexit
import math
import os
import time
from datetime import date

os.environ["HF_HUB_OFFLINE"] = "1"  # models/ 에 받아 둔 파일만 쓴다
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from .preprocess import RESULT_DIR, ROOT, SEED, cv_splits, get_xy

hf_logging.set_verbosity_error()

MODELS = {
    "KcELECTRA-base-v2022": ROOT / "models" / "KcELECTRA-base-v2022",
    "klue-roberta-base": ROOT / "models" / "klue-roberta-base",
}
METHODS = ["frozen", "finetune", "lora"]
CONDITIONS = {"Comment-only": False, "Context+Comment": True}  # 조건 이름 → get_xy 의 use_context
METRICS = ["accuracy", "macro_f1", "recall_hate"]
OUT_CSV = RESULT_DIR / "plm_folds.csv"

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


def texts(df, use_context):
    """get_xy 로 입력을 꺼내 토크나이저에 넣을 형태로 돌려준다: (첫 문장, 둘째 문장 또는 None, 정답)."""
    X, y = get_xy(df, use_context=use_context)
    if use_context:
        return X["Context"].tolist(), X["Comment"].tolist(), y
    return X["Comment"].tolist(), None, y


def encode(tok, first, second, config):
    """문장(또는 문장쌍)을 토큰화한다. 패딩은 배치마다 따로 하므로 여기서는 하지 않는다.

    klue-roberta-base 는 문장 구분 임베딩(type_vocab_size)이 1개뿐이라, 토크나이저가 둘째 문장에 붙이는
    token_type_ids=1 을 넣으면 범위 밖 오류가 난다 → 그런 모델에서는 token_type_ids 를 빼고 [SEP] 로만 구분한다.
    """
    enc = tok(first, second, truncation=True, max_length=MAX_LEN)
    if getattr(config, "type_vocab_size", 2) < 2:
        enc.pop("token_type_ids", None)
    return enc


def batches(enc, idx, tok, device, labels=None):
    """인덱스 순서대로 BATCH 개씩 잘라, 배치 안에서 가장 긴 길이에 맞춰 패딩한 텐서를 낸다."""
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        feats = [{k: enc[k][i] for k in enc.keys()} for i in b]
        out = tok.pad(feats, return_tensors="pt")
        out = {k: v.to(device) for k, v in out.items()}
        if labels is not None:
            out["labels"] = torch.as_tensor(labels[b], device=device)
        yield out


def score(model, method, cond, fold, y_true, y_pred, device, info):
    """fold 하나의 점수와 비용 정보(info: train_sec, sleep_sec, trainable_params, ...)를 한 행으로 저장한다."""
    rows.append({
        "model": model,
        "method": method,
        "condition": cond,
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_hate": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        **info,
        "device": device.type,
    })
    pd.DataFrame(rows).round(6).to_csv(OUT_CSV, index=False)  # 중간에 끊겨도 그때까지 결과가 남도록 매번 저장
    r = rows[-1]
    flag = "  ⚠ 30분 초과" if r["train_sec"] > TIME_LIMIT_SEC else ""
    flag += f"  (잠자기 {r['sleep_sec']:.0f}s 제외)" if r["sleep_sec"] > 5 else ""
    print(f"  [{model} | {method} | {cond} | fold {fold}] macro-F1={r['macro_f1']:.3f} "
          f"recall_hate={r['recall_hate']:.3f} acc={r['accuracy']:.3f} ({r['train_sec']:.0f}s){flag}")


# ───────────────────────────── (a) 고정 임베딩 + LogReg ─────────────────────────────
@torch.no_grad()
def embed_all(model_dir, df, use_context, device):
    """전체 행의 평균 풀링 벡터를 한 번에 구한다. 모델이 고정이라 fold와 무관 → 학습 정보 누설 없음."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc_model = AutoModel.from_pretrained(model_dir).to(device).eval()
    first, second, _ = texts(df, use_context)
    enc = encode(tok, first, second, enc_model.config)
    out = []
    for b in batches(enc, np.arange(len(first)), tok, device):
        h = enc_model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())  # 패딩 제외 평균
    total = sum(p.numel() for p in enc_model.parameters())
    return np.vstack(out), total


def run_frozen(name, folds, device):
    """조건마다 전체 df 임베딩을 한 번 구해 두고, fold마다 그 fold의 train 행으로만 LogReg를 학습한다."""
    full = pd.concat([folds[0][1], folds[0][2]])  # 어느 fold든 train+test = 전체 3000행 (원문 텍스트는 전처리와 무관)
    key = full["Index"].to_numpy()
    pos = pd.Series(np.arange(len(key)), index=key)
    for cond, use_context in CONDITIONS.items():
        with Timer() as t_emb:
            E, total = embed_all(MODELS[name], full, use_context, device)
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
def finetune_one(model_dir, train, test, use_context, device, method="finetune", seed=SEED):
    """fold 하나를 학습하고 (정답, test 예측, 비용 정보) 를 돌려준다. 하이퍼파라미터는 고정값만 쓴다.

    method="finetune": 모델 전체 학습
    method="lora"    : 원래 가중치는 고정, query·value 의 LoRA 행렬과 분류 헤드만 학습
    """
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    if method == "lora":
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA))  # 분류 헤드는 peft 가 학습 대상으로 둔다
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    tr_first, tr_second, y_tr = texts(train, use_context)
    te_first, te_second, y_te = texts(test, use_context)
    enc_tr = encode(tok, tr_first, tr_second, model.config)
    enc_te = encode(tok, te_first, te_second, model.config)

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
    preds = []
    with torch.no_grad():
        for b in batches(enc_te, np.arange(len(y_te)), tok, device):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                preds.append(model(**b).logits.argmax(-1).cpu().numpy())
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
    }
    return y_te, np.concatenate(preds), info


def run_trained(name, folds, device, method):
    for fold, train, test in folds:
        for cond, use_context in CONDITIONS.items():
            y_te, pred, info = finetune_one(MODELS[name], train, test, use_context, device, method=method)
            score(name, method, cond, fold, y_te, pred, device, info)


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
    f1 = df.groupby(["model", "method", "condition"], sort=False)["macro_f1"]
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
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default=None, help="결과 CSV 경로 (기본 results/midterm/reproduced/plm_folds.csv)")
    args = parser.parse_args()
    if args.out:
        OUT_CSV = ROOT / args.out

    atexit.register(log_row)
    set_seed()
    device = pick_device()
    print(f"seed = {SEED}, device = {device}")
    for name in args.models:
        if not (MODELS[name] / "config.json").exists():
            raise FileNotFoundError(f"{MODELS[name]} 가 없습니다. models/ 에 먼저 받아 두세요.")

    if OUT_CSV.exists():
        print(f"알림: {OUT_CSV.relative_to(ROOT)} 이(가) 이미 있어 덮어씁니다.")
    folds = [(f, tr, te) for f, tr, te, _ in cv_splits() if f in args.folds]
    RUN["expected"] = len(args.models) * len(args.methods) * len(CONDITIONS) * len(folds)

    for name in args.models:
        for method in args.methods:
            print(f"\n=== {name} — {method} ===")
            if method == "frozen":
                run_frozen(name, folds, device)
            else:
                run_trained(name, folds, device, method)
    print(f"\n저장: {OUT_CSV.relative_to(ROOT)} ({len(rows)}행)")
