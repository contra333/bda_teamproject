"""KOCOH v.2 기준선(baseline) 실험: Comment-only vs Context + Comment (통합판).

이 파일이 하는 일 (한 줄 요약)
    "preprocess.py 의 5-fold 분할을 그대로 쓰고, 기준선을 같은 fold 에서 평가한다.
     TF-IDF 는 기본 임계값과 'train 안에서 고른 임계값' 두 규칙으로 함께 평가한다."

기준선 10개 조합과 각각이 답하는 질문 (모두 '기본' 규칙으로 평가)
    구분  기준선                 입력 조건                                         실행 수
    바닥  1. Dummy most_frequent  조건 무관                                           1
          Q. 전부 '비혐오'라고 하면 몇 점인가? → 정확도의 바닥 (비혐오가 약 75%)
    바닥  2. Dummy stratified     조건 무관                                           1
          Q. 클래스 비율대로 무작위로 찍으면 몇 점인가? → macro-F1·혐오 Recall 의 바닥
    진단  3. 글자 수만 LogReg      Comment-only / Context+Comment                    2
          Q. 내용을 읽지 않고 길이만으로 맞힐 수 있는가? (지름길 점검)
    전통  4. TF-IDF + LogReg      Comment-only / Context+Comment / Context-only     3
    전통  5. TF-IDF + LinearSVC   Comment-only / Context+Comment / Context-only     3
          Q. 단순한 어휘 모델에서 맥락을 더하면 성능이 오르는가? 모델 종류를 바꿔도 같은가?
          Context-only: 맥락 문장만으로 맞힐 수 있는가? → 맥락 이득이 '주제별 혐오 비율' 에서 오는지 점검
                                                                      합계 10개 조합 × 5 fold

판정 규칙 (rule 열)
    "기본"     : 모델의 predict() 그대로 — LogReg 는 혐오 확률 0.5, LinearSVC 는 결정 함수 값 0
    "OOF 조정" : TF-IDF 두 모델 × 세 조건만. fold 마다 train 을 post_group 단위 4-fold 로 나눠(preprocess.cv_splits_inner)
                 OOF 점수를 만들고, macro-F1 이 가장 높은 임계값을 미리 정한 후보 안에서 고른다(동점이면 기본값에 가까운 쪽).
                 그 뒤 train 전체로 다시 학습한 모델에 그 임계값을 적용한다. test fold 는 고르는 데 쓰지 않는다.
                   후보 — LogReg 확률 0.20~0.80 (0.025 간격) / LinearSVC 결정 함수 −1.0~1.0 (0.05 간격)

같이 기록하는 것
    - PR-AUC (TF-IDF): 임계값과 무관한 지표. 무작위 기준선 = test 혐오 비율(약 0.25)
    - Type별 정답률 (TF-IDF): 예측이 끝난 뒤 채점 분류용으로만 Type 사용 (입력 아님). 'Type 1·2 짝' = 같은 댓글의 Type 1·2 행을 모두 맞힌 비율
    - Comment-only 이론적 상한: Type 1·2 는 같은 댓글에 정답만 반대라, 같은 입력에 같은 답을 내는 모델은 짝 중 하나를 반드시 틀린다.
      test 안에서 '같은 댓글 중 하나라도 혐오면 혐오' 로 답하는 것이 최선이며, 그 점수를 fold 별로 잰다.

뺀 것
    - 보조 라벨(Profanity 등)로 만든 규칙: 정답 누설이라 쓰지 않는다.

지켜야 할 점
    - 벡터화·학습은 각 fold 의 train 으로만 fit 한다 (test 는 transform 만).
    - 모델 입력은 get_xy() 로만 꺼낸다 (누설 열이 섞이지 않도록).
    - Context+Comment 는 Context·Comment 를 각각 별도 TF-IDF 로 바꿔 옆으로 붙인다(hstack).
      Comment 쪽 설정은 Comment-only 와 같으므로, 두 조건의 차이는 'Context 특성 추가' 하나뿐이다.

결과 파일 (results/)
    baseline_folds.csv         : 기본 10조합 × 5 fold = 50행 + OOF 조정 6조합 × 5 fold = 30행 → 80행
    baseline_by_type.csv       : TF-IDF 2모델 × 3조건 × 2규칙 × 5 fold × 4그룹 = 240행
    comment_only_ceiling.csv   : 5 fold = 5행
    split_assignment.csv, inner_val_composition.csv  (preprocess.cv_splits_inner 가 저장)

터미널에서 실행
    python src/baseline.py
"""
import atexit
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

from preprocess import DATA_PATH, ROOT, SEED, TARGET, TEXT_COLS, cv_splits_inner, get_xy

OUT_CSV = ROOT / "results" / "baseline_folds.csv"
OUT_TYPE = ROOT / "results" / "baseline_by_type.csv"
OUT_CEIL = ROOT / "results" / "comment_only_ceiling.csv"

CONDITIONS = {"Comment-only": False, "Context+Comment": True}  # 글자 수 모델용: 조건 이름 → get_xy 의 use_context
TFIDF_CONDITIONS = {"Comment-only": ["Comment"], "Context+Comment": ["Context", "Comment"], "Context-only": ["Context"]}
METRICS = ["accuracy", "macro_f1", "recall_hate"]
EXTRA_METRICS = ["precision_hate", "pr_auc"]
RULES = ["기본", "OOF 조정"]
N_EXPECTED = 80  # 기본 10조합 × 5 + OOF 조정 6조합 × 5 — 이만큼 모여야 실험로그를 남긴다

rows = []  # fold별 점수
type_rows = []  # Type별 정답률
ceil_rows = []  # Comment-only 상한
summary = {}  # 실험로그용 지표 문자열 (끝까지 돌았을 때만 채워짐)


# ───────────────────────────── 1. 모델 만들기 ─────────────────────────────
def make_tfidf():
    """글자 n-gram TF-IDF. 한국어는 어미·조사 변화가 많아 단어보다 글자 단위가 안정적이다."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)


def make_logreg():
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)


def make_svc():
    return LinearSVC(class_weight="balanced", random_state=SEED)


# 모델 이름 → (만들기, 점수 종류, 임계값 후보, 기본 임계값)
TFIDF_MODELS = {
    "TF-IDF LogReg": (make_logreg, "proba", np.round(np.arange(0.20, 0.8001, 0.025), 3), 0.5),
    "TF-IDF LinearSVC": (make_svc, "margin", np.round(np.arange(-1.0, 1.0001, 0.05), 2), 0.0),
}


# ───────────────────────────── 2. 특성 만들기 ─────────────────────────────
def length_features(train, test, use_context):
    """get_xy(use_length=True) 에서 숫자 열(글자 수)만 꺼낸다. 표준화는 cv_splits 가 fold별로 이미 함."""
    X_tr, y_tr = get_xy(train, use_context=use_context, use_length=True)
    X_te, y_te = get_xy(test, use_context=use_context, use_length=True)
    num_cols = [c for c in X_tr.columns if c not in TEXT_COLS]
    return X_tr[num_cols].to_numpy(), y_tr, X_te[num_cols].to_numpy(), y_te


def tfidf_features(train, test, cols):
    """텍스트 열마다 별도 TF-IDF를 train으로 fit 하고, 열별 결과를 옆으로 붙인다 (cols 는 get_xy 허용 열 안에서)."""
    X_tr, y_tr = get_xy(train, use_context=True)
    X_te, y_te = get_xy(test, use_context=True)
    tr_parts, te_parts = [], []
    for col in cols:  # Comment-only: [Comment] / Context+Comment: [Context, Comment] / Context-only: [Context]
        vec = make_tfidf().fit(X_tr[col])
        tr_parts.append(vec.transform(X_tr[col]))
        te_parts.append(vec.transform(X_te[col]))
    return hstack(tr_parts).tocsr(), y_tr, hstack(te_parts).tocsr(), y_te


# ───────────────────────────── 3. 점수와 임계값 ─────────────────────────────
def fit_score(make, kind, train, test, cols):
    """train 으로 TF-IDF·모델을 fit 하고 test 의 '혐오 쪽 점수' 를 돌려준다 (확률 또는 결정 함수 값)."""
    X_tr, y_tr, X_te, _ = tfidf_features(train, test, cols)
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


# ───────────────────────────── 4. 평가·기록 ─────────────────────────────
def score(model, cond, fold, y_true, y_pred, rule="기본", threshold=np.nan, y_score=None):
    rows.append({
        "model": model,
        "condition": cond,
        "rule": rule,
        "threshold": threshold,
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_hate": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_hate": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score) if y_score is not None else np.nan,
    })


def by_type(model, cond, rule, fold, types, comments, y_true, y_pred):
    """예측이 끝난 뒤 Type 별로 정답률·혐오 예측 비율을 센다 (Type 은 입력에 쓰지 않음)."""
    correct = y_pred == y_true
    for t in (1, 2, 3):
        m = types == t
        type_rows.append({"model": model, "condition": cond, "rule": rule, "fold": fold, "group": f"Type {t}",
                          "n": int(m.sum()), "accuracy": correct[m].mean(), "pred_hate_rate": y_pred[m].mean()})
    # 1·2 짝: test 안에 Type 1 과 Type 2 가 모두 있는 댓글마다 '두 유형의 행을 전부 맞혔는가'
    d = pd.DataFrame({"comment": comments, "type": types, "correct": correct})
    d = d[d["type"].isin([1, 2])]
    both = d.groupby("comment")["type"].nunique() == 2
    pair_ok = d[d["comment"].isin(both[both].index)].groupby("comment")["correct"].all()
    type_rows.append({"model": model, "condition": cond, "rule": rule, "fold": fold, "group": "Type 1·2 짝",
                      "n": int(len(pair_ok)), "accuracy": pair_ok.mean() if len(pair_ok) else np.nan,
                      "pred_hate_rate": np.nan})


def comment_only_ceiling(fold, test):
    """같은 댓글에는 같은 답을 내는 모델의 최선: 그 댓글이 test 에서 한 번이라도 혐오면 혐오로 답한다."""
    y = test[TARGET].to_numpy()
    best = test.groupby("Comment")[TARGET].transform("max").to_numpy()
    ceil_rows.append({"fold": fold, "accuracy": accuracy_score(y, best),
                      "macro_f1": f1_score(y, best, average="macro", zero_division=0),
                      "recall_hate": recall_score(y, best, pos_label=1, zero_division=0)})


def run_fold(fold, train, test, inner=None, type_of=None):
    """fold 하나를 평가한다.

    inner   : preprocess.cv_splits_inner 의 내부 fold. 주면 TF-IDF 에 'OOF 조정' 규칙을 추가로 평가한다.
    type_of : Index → Type 표. 주면 Type별 정답률과 Comment-only 상한을 기록한다 (채점 분류용).
    둘 다 없으면 '기본' 규칙 10개 조합만 평가한다 (노트북에서 빠르게 확인할 때).
    """
    # (1)(2) Dummy: 입력을 보지 않으므로 조건 구분 없이 1회
    _, y_tr = get_xy(train, use_context=False)
    _, y_te = get_xy(test, use_context=False)
    for strategy in ("most_frequent", "stratified"):
        clf = DummyClassifier(strategy=strategy, random_state=SEED)
        clf.fit(np.zeros((len(y_tr), 1)), y_tr)
        score(f"Dummy {strategy}", "-", fold, y_te, clf.predict(np.zeros((len(y_te), 1))))

    # (3) 글자 수만 LogReg
    for cond, use_context in CONDITIONS.items():
        X_tr, y_tr, X_te, y_te = length_features(train, test, use_context)
        score("Length LogReg", cond, fold, y_te, make_logreg().fit(X_tr, y_tr).predict(X_te))

    # (4)(5) TF-IDF + 선형 모델 — 기본 규칙, 그리고 (inner 가 있으면) OOF 조정 규칙
    types = test["Index"].map(type_of).to_numpy() if type_of is not None else None
    comments = test["Comment"].to_numpy()
    for cond, cols in TFIDF_CONDITIONS.items():
        for model, (make, kind, grid, default) in TFIDF_MODELS.items():
            s = fit_score(make, kind, train, test, cols)  # train 전체로 fit
            rules = [(RULES[0], default)]
            if inner is not None:
                rules.append((RULES[1], choose_threshold(y_tr, oof_scores(make, kind, train, cols, inner), grid, default)))
            for rule, th in rules:
                pred = (s > th).astype(int)  # sklearn predict 와 같은 '초과' 규칙 → 기본 규칙은 predict() 와 정확히 같다
                score(model, cond, fold, y_te, pred, rule=rule, threshold=th, y_score=s)
                if types is not None:
                    by_type(model, cond, rule, fold, types, comments, y_te, pred)

    if type_of is not None:
        comment_only_ceiling(fold, test)


# ───────────────────────────── 5. 요약·판정 ─────────────────────────────
def mean_std(s):
    """평균 ± 표준편차(표본, ddof=1) 문자열."""
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"


def context_gain(df, metrics=METRICS):
    """모델·규칙별 맥락 이득 Δ = (Context+Comment) − (Comment-only). 같은 fold끼리 짝지어 뺀다."""
    d = df[df["condition"].isin(["Comment-only", "Context+Comment"])]
    paired = d.pivot_table(index=["model", "rule", "fold"], columns="condition", values=list(metrics))
    gain = pd.DataFrame({m: paired[(m, "Context+Comment")] - paired[(m, "Comment-only")] for m in metrics})
    return gain.groupby(level=["model", "rule"], sort=False).agg(mean_std)


def verdict(d):
    """노트북 2-6절 판정 규칙: 평균 Δ 부호 + 5개 중 4개 이상 같은 방향 + 대응 t-검정 p < 0.05."""
    p = float(stats.ttest_rel(d, np.zeros_like(d)).pvalue) if np.any(d != 0) else 1.0
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    if d.mean() > 0 and pos >= 4 and p < 0.05:
        v = "도움 됨"
    elif d.mean() < 0 and neg >= 4 and p < 0.05:
        v = "해가 됨"
    else:
        v = "판단 보류"
    return {"평균 Δ": f"{d.mean():+.3f} ± {d.std(ddof=1):.3f}", "Δ>0 fold": f"{pos}/{len(d)}",
            "p": "<0.001" if p < 0.001 else f"{p:.3f}", "판정": v}


def context_gain_verdict(df):
    """TF-IDF 모델·규칙별로 macro-F1·혐오 Recall·PR-AUC 의 맥락 이득을 판정한다."""
    out = {}
    tf = df[df["model"].isin(TFIDF_MODELS)]
    for (model, rule), g in tf.groupby(["model", "rule"], sort=False):
        w = g.pivot(index="fold", columns="condition", values=["macro_f1", "recall_hate", "pr_auc"])
        for m in ["macro_f1", "recall_hate", "pr_auc"]:
            out[(model, rule, m)] = verdict((w[(m, "Context+Comment")] - w[(m, "Comment-only")]).to_numpy())
    res = pd.DataFrame(out).T
    res.index.names = ["모델", "규칙", "지표"]
    return res


# ───────────────────────────── 6. 실험로그 (atexit) ─────────────────────────────
def log_row():
    """끝까지 돌았을 때만 실험로그 한 행을 출력한다. 중간에 실패하면 행을 만들지 않는다."""
    if len(rows) != N_EXPECTED or not summary:
        print(f"\n결과 미완 — 로그 생략 (모인 결과 {len(rows)} / {N_EXPECTED})")
        return
    config = f"split=post_group 5-fold StratifiedGroupKFold, seed={SEED}, 임계값=train 내부 4-fold OOF"
    result_file = OUT_CSV.relative_to(ROOT)
    print("\n[실험로그]")
    print("| 실험 이름 | 설정 | 지표 | 결과 파일 | 날짜 |")
    print("|---|---|---|---|---|")
    print(f"| baseline | {config} | {summary['metrics']} | {result_file} | {date.today().isoformat()} |")


# ───────────────────────────── 터미널에서 직접 실행할 때 ─────────────────────────────
if __name__ == "__main__":
    atexit.register(log_row)
    np.random.seed(SEED)
    print(f"seed = {SEED}")
    type_of = pd.read_csv(DATA_PATH).set_index("Index")["Type"]  # 채점 분류용 (입력 아님)

    for fold, train, test, inner in cv_splits_inner():
        run_fold(fold, train, test, inner=inner, type_of=type_of)

    df, by_t, ceil = pd.DataFrame(rows), pd.DataFrame(type_rows), pd.DataFrame(ceil_rows)

    print("\n[기준선 5-fold 평균 ± 표준편차]")
    print(df.groupby(["model", "condition", "rule"], sort=False)[METRICS + EXTRA_METRICS].agg(mean_std).to_string())

    print("\n[고른 임계값 — fold별]")
    print(df[df["rule"] == RULES[1]].pivot_table(index=["model", "condition"], columns="fold",
                                                   values="threshold", sort=False).to_string())

    print("\n[맥락 이득 Δ = (Context+Comment) − (Comment-only), 같은 fold끼리]")
    print(context_gain_verdict(df).to_string())

    print("\n[Type별 정답률 — 5-fold 평균] (Type 은 채점 분류용, 입력 아님)")
    print(by_t.groupby(["model", "condition", "rule", "group"], sort=False)["accuracy"].mean().unstack("group")
          .round(3).to_string())

    print("\n[Comment-only 이론적 상한 — 5-fold 평균 ± 표준편차]")
    print(ceil[METRICS].agg(mean_std).to_string())

    for out, d in [(OUT_CSV, df), (OUT_TYPE, by_t), (OUT_CEIL, ceil)]:
        d.round(6).to_csv(out, index=False)
    print(f"\n저장: {OUT_CSV.relative_to(ROOT)} ({len(df)}행), {OUT_TYPE.relative_to(ROOT)} ({len(by_t)}행), "
          f"{OUT_CEIL.relative_to(ROOT)} ({len(ceil)}행)")

    f1 = df.groupby(["model", "condition", "rule"], sort=False)["macro_f1"]
    summary["metrics"] = ", ".join(f"macro_f1[{m} / {c} / {r}]={mean_std(s)}" for (m, c, r), s in f1)
