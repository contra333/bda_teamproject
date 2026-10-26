"""KOCOH v.2 기준선 v2: baseline.py 의 8개 조합 + 중간보고서에 쓴 진단 3가지.

이 파일이 하는 일 (한 줄 요약)
    "baseline.py 를 그대로 돌리고, 중간보고서 5장에 인용한 진단 숫자를 같은 5-fold 에서 재현해 저장한다."

baseline.py 와 달라진 점 (원본 baseline.py·results/midterm/reproduced/baseline_folds.csv 는 건드리지 않음)
    1. Context-only 조건 추가 (TF-IDF LogReg · LinearSVC)
       Q. 맥락 문장만으로 혐오를 맞힐 수 있는가?
          → 맥락 이득이 '맥락–댓글 관계'가 아니라 '게시글 주제별 혐오 비율'에서 오는지 점검한다.
            처음 보는 게시글(test 묶음)에서 이 점수가 바닥선 수준이면 주제 단서만으로는 맞힐 수 없다.
    2. Type별 정답률 (TF-IDF 두 모델 × 세 조건)
       Q. 맥락을 넣었을 때 어느 유형에서 맞고 틀리는가?
          → 혐오 Recall 이 떨어지는 이유를 본다. Type 1 = 혐오, Type 2 = 사람이 만든 맥락 + 같은 댓글(비혐오),
            Type 3 = 실제 맥락 + 다른 댓글(비혐오). '1·2 짝' = 같은 댓글의 Type 1·2 행을 모두 맞힌 비율.
          ※ Type 은 정답을 100% 결정하므로 입력에는 절대 넣지 않는다. 예측이 끝난 뒤 '채점 분류용'으로만 쓴다.
    3. Comment-only 이론적 상한 (모델 없음)
       Q. 댓글만 보는 모델이 낼 수 있는 최고 점수는?
          → Type 1·2 는 같은 댓글에 정답만 반대라, 같은 입력에는 같은 예측을 내는 모델은 짝 중 하나를 반드시 틀린다.
            test 안에서 '같은 댓글 중 하나라도 혐오면 혐오'로 답하는 것이 최선이며, 그 점수를 fold별로 잰다.

결과 파일 (모두 새 이름 — 기존 결과를 덮어쓰지 않음)
    results/midterm/reproduced/baseline_v2_folds.csv       : 10개 조합 × 5 fold = 50행 (앞 8개 조합은 baseline_folds.csv 와 같은 값)
    results/midterm/reproduced/baseline_v2_by_type.csv     : TF-IDF 2모델 × 3조건 × 5 fold × 4그룹(Type 1·2·3, 1·2 짝) = 120행
    results/midterm/reproduced/comment_only_ceiling.csv    : 5 fold = 5행

터미널에서 실행
    python -m src.midterm.baseline_v2
"""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score, recall_score

from . import baseline as bl
from .preprocess import RESULT_DIR, DATA_PATH, ROOT, SEED, TARGET, cv_splits, get_xy

OUT_FOLDS = RESULT_DIR / "baseline_v2_folds.csv"
OUT_TYPE = RESULT_DIR / "baseline_v2_by_type.csv"
OUT_CEIL = RESULT_DIR / "comment_only_ceiling.csv"

# 조건 이름 → 입력 열 (모두 get_xy(use_context=True) 가 돌려준 허용 열 안에서 고른다)
CONDITIONS = {"Comment-only": ["Comment"], "Context+Comment": ["Context", "Comment"], "Context-only": ["Context"]}
TFIDF_MODELS = {"TF-IDF LogReg": bl.make_logreg, "TF-IDF LinearSVC": bl.make_svc}

type_rows = []  # Type별 정답률
ceil_rows = []  # Comment-only 상한


# ───────────────────────────── 1. 특성 ─────────────────────────────
def tfidf_features(train, test, cols):
    """baseline.tfidf_features 와 같은 방식(열마다 별도 TF-IDF, train 으로만 fit)으로 원하는 열만 쓴다."""
    X_tr, y_tr = get_xy(train, use_context=True)
    X_te, y_te = get_xy(test, use_context=True)
    tr_parts, te_parts = [], []
    for col in cols:
        vec = bl.make_tfidf().fit(X_tr[col])
        tr_parts.append(vec.transform(X_tr[col]))
        te_parts.append(vec.transform(X_te[col]))
    return hstack(tr_parts).tocsr(), y_tr, hstack(te_parts).tocsr(), y_te


# ───────────────────────────── 2. 진단 ─────────────────────────────
def by_type(model, cond, fold, types, comments, y_true, y_pred):
    """예측이 끝난 뒤 Type 별로 정답률·혐오 예측 비율을 센다 (Type 은 입력에 쓰지 않음)."""
    correct = y_pred == y_true
    for t in (1, 2, 3):
        m = types == t
        type_rows.append({"model": model, "condition": cond, "fold": fold, "group": f"Type {t}", "n": int(m.sum()),
                          "accuracy": correct[m].mean(), "pred_hate_rate": y_pred[m].mean()})
    # 1·2 짝: test 안에 Type 1 과 Type 2 가 모두 있는 댓글마다 '두 유형의 행을 전부 맞혔는가'
    d = pd.DataFrame({"comment": comments, "type": types, "correct": correct})
    d = d[d["type"].isin([1, 2])]
    both = d.groupby("comment")["type"].nunique() == 2
    pair_ok = d[d["comment"].isin(both[both].index)].groupby("comment")["correct"].all()
    type_rows.append({"model": model, "condition": cond, "fold": fold, "group": "Type 1·2 짝", "n": int(len(pair_ok)),
                      "accuracy": pair_ok.mean() if len(pair_ok) else np.nan, "pred_hate_rate": np.nan})


def comment_only_ceiling(fold, test):
    """같은 댓글에는 같은 답을 내는 모델의 최선: 그 댓글이 test 에서 한 번이라도 혐오면 혐오로 답한다."""
    y = test[TARGET].to_numpy()
    best = test.groupby("Comment")[TARGET].transform("max").to_numpy()
    ceil_rows.append({"fold": fold, "accuracy": accuracy_score(y, best),
                      "macro_f1": f1_score(y, best, average="macro", zero_division=0),
                      "recall_hate": recall_score(y, best, pos_label=1, zero_division=0)})


# ───────────────────────────── 3. fold 하나 ─────────────────────────────
def run_fold(fold, train, test, type_of):
    bl.run_fold(fold, train, test)  # 기존 8개 조합 (baseline.py 와 같은 코드·같은 값)

    types = test["Index"].map(type_of).to_numpy()
    comments = test["Comment"].to_numpy()
    for cond, cols in CONDITIONS.items():
        X_tr, y_tr, X_te, y_te = tfidf_features(train, test, cols)
        for model, make in TFIDF_MODELS.items():
            pred = make().fit(X_tr, y_tr).predict(X_te)
            if cond == "Context-only":  # 나머지 두 조건의 점수는 bl.run_fold 가 이미 기록함
                bl.score(model, cond, fold, y_te, pred)
            by_type(model, cond, fold, types, comments, y_te, pred)
    comment_only_ceiling(fold, test)


# ───────────────────────────── 터미널에서 직접 실행할 때 ─────────────────────────────
if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    print(f"seed = {SEED}")
    # Type 은 전처리에서 지워지므로 원본에서 Index → Type 표만 따로 만든다 (채점 분류용)
    type_of = pd.read_csv(DATA_PATH).set_index("Index")["Type"]

    bl.rows.clear()
    for fold, train, test, _ in cv_splits():
        run_fold(fold, train, test, type_of)

    df = pd.DataFrame(bl.rows)
    by_t = pd.DataFrame(type_rows)
    ceil = pd.DataFrame(ceil_rows)

    # 기존 8개 조합이 results/midterm/reproduced/baseline_folds.csv 와 같은지 확인 (다르면 저장하지 않고 멈춤)
    saved = pd.read_csv(bl.OUT_CSV)
    same = df[df["condition"] != "Context-only"].reset_index(drop=True)
    assert np.allclose(same[bl.METRICS].to_numpy(), saved[bl.METRICS].to_numpy(), atol=1e-6), \
        "기존 8개 조합이 baseline_folds.csv 와 다름 — 데이터·라이브러리 버전 확인"
    print("\n기존 8개 조합: results/midterm/reproduced/baseline_folds.csv 와 동일")

    print("\n[기준선 v2 — 5-fold 평균 ± 표준편차]")
    print(df.groupby(["model", "condition"], sort=False)[bl.METRICS].agg(bl.mean_std).to_string())

    print("\n[Type별 정답률 — 5-fold 평균] (Type 은 채점 분류용, 입력 아님)")
    print(by_t.groupby(["model", "condition", "group"], sort=False)[["accuracy", "pred_hate_rate"]]
          .mean().round(3).to_string())

    print("\n[Comment-only 이론적 상한 — 5-fold 평균 ± 표준편차]")
    print(ceil[bl.METRICS].agg(bl.mean_std).to_string())

    df.round(6).to_csv(OUT_FOLDS, index=False)
    by_t.round(6).to_csv(OUT_TYPE, index=False)
    ceil.round(6).to_csv(OUT_CEIL, index=False)
    print(f"\n저장: {OUT_FOLDS.relative_to(ROOT)} ({len(df)}행), {OUT_TYPE.relative_to(ROOT)} ({len(by_t)}행), "
          f"{OUT_CEIL.relative_to(ROOT)} ({len(ceil)}행)")
