"""KOCOH v.2 기준선 v3: TF-IDF 판정 임계값을 train 안에서만 골라 0.5 고정과 비교한다.

이 파일이 하는 일 (한 줄 요약)
    "fold 마다 train 안의 묶음 단위 4-fold 로 예측(OOF)을 만들어 임계값을 고르고,
     train 전체로 다시 학습한 모델에 그 임계값을 적용해 test 에서 평가한다."

절차 (TF-IDF 두 모델 × 세 조건, fold 마다 똑같이)
    1. 선택: train 을 post_group 단위로 4등분 → 3조각으로 TF-IDF·모델 fit, 남은 조각 예측 → 4번 반복해 OOF 점수
    2. OOF 점수에서 macro-F1 이 가장 높은 임계값을 '미리 정한 후보' 안에서 고른다 (동점이면 기본값에 가까운 쪽)
         LogReg   : 혐오 확률, 후보 0.20 ~ 0.80 (0.025 간격), 기본값 0.5
         LinearSVC: 결정 함수 값, 후보 −1.0 ~ 1.0 (0.05 간격), 기본값 0  (확률이 없으므로)
    3. 최종: 같은 설정으로 train 전체에 다시 fit → test 점수에 고른 임계값 적용
    test fold 는 1·2 단계에 전혀 쓰지 않는다. 두 조건(Comment-only / Context+Comment / Context-only)에 같은 절차.

같이 기록하는 것
    - '0.5 고정'(LinearSVC 는 0) 결과 — baseline_v2 와 같은 값인지 확인 (다르면 멈춤)
    - PR-AUC — 임계값과 무관한 지표 (혐오를 위쪽에 놓는 능력). 무작위 기준선 = test 혐오 비율(약 0.25)
    - Type별 정답률 — 예측이 끝난 뒤 채점 분류용으로만 Type 사용 (입력 아님)
    - 맥락 이득 Δ 판정 — 노트북 2-6절 규칙: 평균 Δ 부호 + 5개 중 4개 이상 같은 방향 + 대응 t-검정 p < 0.05

결과 파일 (모두 새 이름)
    results/midterm/reproduced/baseline_v3_folds.csv    : 2모델 × 3조건 × 2규칙 × 5 fold = 60행
    results/midterm/reproduced/baseline_v3_by_type.csv  : 위 × 4그룹 = 240행
    results/midterm/reproduced/split_assignment_v2.csv, results/midterm/reproduced/inner_val_composition.csv  (preprocess_v2.py 가 저장)

터미널에서 실행
    python -m src.midterm.baseline_v3
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score

from . import baseline as bl
from . import baseline_v2 as b2
from .preprocess import RESULT_DIR, DATA_PATH, ROOT, SEED
from .preprocess_v2 import cv_splits_v2

OUT_FOLDS = RESULT_DIR / "baseline_v3_folds.csv"
OUT_TYPE = RESULT_DIR / "baseline_v3_by_type.csv"

# 모델 이름 → (만들기, 점수 종류, 임계값 후보, 기본 임계값)
MODELS = {
    "TF-IDF LogReg": (bl.make_logreg, "proba", np.round(np.arange(0.20, 0.8001, 0.025), 3), 0.5),
    "TF-IDF LinearSVC": (bl.make_svc, "margin", np.round(np.arange(-1.0, 1.0001, 0.05), 2), 0.0),
}
METRICS = ["accuracy", "macro_f1", "recall_hate", "precision_hate"]
RULES = ["0.5 고정", "OOF 조정"]

rows, type_rows = [], []


# ───────────────────────────── 1. 점수와 임계값 ─────────────────────────────
def fit_score(make, kind, train, test, cols):
    """train 으로 TF-IDF·모델을 fit 하고 test 의 '혐오 쪽 점수' 를 돌려준다 (확률 또는 결정 함수 값)."""
    X_tr, y_tr, X_te, _ = b2.tfidf_features(train, test, cols)
    model = make().fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1] if kind == "proba" else model.decision_function(X_te)


def oof_scores(make, kind, train, cols, inner):
    """train 안의 묶음 단위 내부 fold 로 OOF 점수를 만든다. 각 행은 자기 묶음을 보지 않은 모델의 점수를 받는다."""
    s = np.zeros(len(train))
    for itr, iva in inner:
        s[iva] = fit_score(make, kind, train.iloc[itr], train.iloc[iva], cols)
    return s


def choose_threshold(y, s, grid, default):
    """후보 중 macro-F1 최대인 임계값. 동점이면 기본값에 가장 가까운 것 (불필요하게 움직이지 않도록)."""
    f1 = np.array([f1_score(y, (s > t).astype(int), average="macro", zero_division=0) for t in grid])
    best = np.flatnonzero(np.isclose(f1, f1.max()))
    return float(grid[best[np.argmin(np.abs(grid[best] - default))]])


# ───────────────────────────── 2. 기록 ─────────────────────────────
def record(model, cond, fold, rule, th, y, s, types, comments):
    pred = (s > th).astype(int)  # sklearn predict 와 같은 '초과' 규칙 → 고정 규칙이 기존 결과와 정확히 일치
    rows.append({"model": model, "condition": cond, "fold": fold, "rule": rule, "threshold": th,
                 "accuracy": accuracy_score(y, pred),
                 "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                 "recall_hate": recall_score(y, pred, zero_division=0),
                 "precision_hate": precision_score(y, pred, zero_division=0),
                 "pr_auc": average_precision_score(y, s)})
    b2.type_rows.clear()
    b2.by_type(model, cond, fold, types, comments, y, pred)
    type_rows.extend({**r, "rule": rule} for r in b2.type_rows)


def run_fold(fold, train, test, inner, type_of):
    y_tr, y_te = train["Hate speech"].to_numpy(), test["Hate speech"].to_numpy()
    types, comments = test["Index"].map(type_of).to_numpy(), test["Comment"].to_numpy()
    for cond, cols in b2.CONDITIONS.items():
        for model, (make, kind, grid, default) in MODELS.items():
            th = choose_threshold(y_tr, oof_scores(make, kind, train, cols, inner), grid, default)
            s = fit_score(make, kind, train, test, cols)  # 최종: train 전체로 다시 fit
            record(model, cond, fold, RULES[0], default, y_te, s, types, comments)
            record(model, cond, fold, RULES[1], th, y_te, s, types, comments)


# ───────────────────────────── 3. 판정 (노트북 2-6절 규칙) ─────────────────────────────
def verdict(d):
    p = float(stats.ttest_rel(d, np.zeros_like(d)).pvalue) if np.any(d != 0) else 1.0
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    if d.mean() > 0 and pos >= 4 and p < 0.05:
        v = "도움 됨"
    elif d.mean() < 0 and neg >= 4 and p < 0.05:
        v = "해가 됨"
    else:
        v = "판단 보류"
    return {"평균 Δ": f"{d.mean():+.3f} ± {d.std(ddof=1):.3f}", "Δ>0 fold": f"{pos}/5",
            "p": "<0.001" if p < 0.001 else f"{p:.3f}", "판정": v}


def context_gain(df):
    out = {}
    for (model, rule), g in df.groupby(["model", "rule"], sort=False):
        w = g.pivot(index="fold", columns="condition", values=["macro_f1", "recall_hate", "pr_auc"])
        for m in ["macro_f1", "recall_hate", "pr_auc"]:
            out[(model, rule, m)] = verdict((w[(m, "Context+Comment")] - w[(m, "Comment-only")]).to_numpy())
    res = pd.DataFrame(out).T
    res.index.names = ["모델", "규칙", "지표"]
    return res


# ───────────────────────────── 터미널에서 직접 실행할 때 ─────────────────────────────
if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    print(f"seed = {SEED}")
    type_of = pd.read_csv(DATA_PATH).set_index("Index")["Type"]  # 채점 분류용 (입력 아님)

    for fold, train, test, inner in cv_splits_v2():
        run_fold(fold, train, test, inner, type_of)
    df, by_t = pd.DataFrame(rows), pd.DataFrame(type_rows)

    # '0.5 고정' 이 baseline_v2 결과와 같은지 확인 (다르면 저장하지 않고 멈춤)
    v2 = pd.read_csv(b2.OUT_FOLDS).set_index(["model", "condition", "fold"])
    fixed = df[df["rule"] == RULES[0]].set_index(["model", "condition", "fold"])
    ref = v2.loc[fixed.index, bl.METRICS]
    assert np.allclose(fixed[bl.METRICS].to_numpy(), ref.to_numpy(), atol=1e-6), "0.5 고정 결과가 baseline_v2 와 다름"
    print("\n'0.5 고정' 결과: results/midterm/reproduced/baseline_v2_folds.csv 와 동일")

    print("\n[고른 임계값 — fold별]")
    print(df[df["rule"] == RULES[1]].pivot_table(index=["model", "condition"], columns="fold",
                                                   values="threshold", sort=False).to_string())

    print("\n[5-fold 평균 ± 표준편차]")
    print(df.groupby(["model", "condition", "rule"], sort=False)[METRICS + ["pr_auc"]].agg(bl.mean_std).to_string())

    print("\n[맥락 이득 Δ = (Context+Comment) − (Comment-only), 같은 fold끼리]")
    print(context_gain(df).to_string())

    print("\n[Type별 정답률 — 5-fold 평균] (Type 은 채점 분류용, 입력 아님)")
    print(by_t[by_t["condition"] != "Context-only"]
          .groupby(["model", "condition", "rule", "group"], sort=False)["accuracy"].mean().unstack("group")
          .round(3).to_string())

    df.round(6).to_csv(OUT_FOLDS, index=False)
    by_t.round(6).to_csv(OUT_TYPE, index=False)
    print(f"\n저장: {OUT_FOLDS.relative_to(ROOT)} ({len(df)}행), {OUT_TYPE.relative_to(ROOT)} ({len(by_t)}행)")
