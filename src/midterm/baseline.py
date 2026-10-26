"""KOCOH v.2 기준선(baseline) 실험: Comment-only vs Context + Comment.

이 파일이 하는 일 (한 줄 요약)
    "preprocess.py 의 5-fold 분할(cv_splits)을 그대로 쓰고, 기준선 8개 조합을 같은 fold에서 평가한다."

기준선 8개 조합과 각각이 답하는 질문
    구분  기준선                 입력 조건                          실행 수
    바닥  1. Dummy most_frequent  조건 무관 (입력을 보지 않음)         1
          Q. 전부 '비혐오'라고 하면 몇 점인가?
             → 정확도의 바닥. 비혐오가 약 75%라 정확도만 보면 아무것도 안 배운 모델도 높게 나온다.
    바닥  2. Dummy stratified     조건 무관 (입력을 보지 않음)         1
          Q. 클래스 비율대로 무작위로 찍으면 몇 점인가?
             → macro-F1·혐오 Recall의 바닥. most_frequent 는 혐오를 한 번도 예측하지 않아
               Recall이 0이므로, '찍기'보다 나은지는 이쪽과 비교한다.
    진단  3. 글자 수만 LogReg      Comment-only / Context+Comment     2
          Q. 내용을 읽지 않고 텍스트 길이라는 표면 특징만으로 라벨을 맞힐 수 있는가?
             → 데이터를 만드는 과정에서 생긴 길이 차이가 지름길이 되는지 확인한다.
               이 점수가 높으면 아래 텍스트 모델의 점수 일부도 길이 덕일 수 있다.
    전통  4. TF-IDF + LogReg      Comment-only / Context+Comment     2
          Q. 단순한 어휘 모델에서 게시글 맥락을 더하면 성능이 오르는가?
             → 프로젝트 핵심 비교의 전통 모델 기준선. 이후 사전학습 모델의 맥락 이득과 견준다.
    전통  5. TF-IDF + LinearSVC   Comment-only / Context+Comment     2
          Q. 선형 모델 종류를 바꿔도 같은 결론이 나오는가?
             → 4번의 결론이 특정 모델에만 해당하는 것이 아닌지 확인한다.
                                                             합계 8개 조합 × 5 fold = 40번 학습

    뺀 것
    - Context-only: 범위에서 제외. 그래서 맥락 이득이 '맥락 이해'에서 온 것인지
      '게시글 주제별 혐오 비율'에서 온 것인지는 구분하지 못한다 (결과 해석 시 한계로 적을 것).
    - 보조 라벨(Profanity 등)로 만든 규칙: 정답 누설이라 쓰지 않는다.

지켜야 할 점
    - 벡터화·학습은 각 fold의 train으로만 fit 한다 (test는 transform만).
    - 모델 입력은 get_xy()로만 꺼낸다 (누설 열이 섞이지 않도록).
    - Context+Comment 는 Context·Comment 를 각각 별도 TF-IDF로 바꿔 옆으로 붙인다(hstack).
      Comment 쪽 설정은 Comment-only 와 같으므로, 두 조건의 차이는 'Context 특성 추가' 하나뿐이다.

터미널에서 실행
    python -m src.midterm.baseline
"""
import atexit
from datetime import date

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import LinearSVC

from .preprocess import RESULT_DIR, ROOT, SEED, TEXT_COLS, cv_splits, get_xy

OUT_CSV = RESULT_DIR / "baseline_folds.csv"
CONDITIONS = {"Comment-only": False, "Context+Comment": True}  # 조건 이름 → get_xy 의 use_context
METRICS = ["accuracy", "macro_f1", "recall_hate"]
N_EXPECTED = 8 * 5  # 8개 조합 × 5 fold — 이만큼 모여야 실험로그를 남긴다

rows = []  # fold별 점수 (model, condition, fold, 지표 3개)
summary = {}  # 실험로그용 지표 문자열 (끝까지 돌았을 때만 채워짐)


# ───────────────────────────── 1. 모델 만들기 ─────────────────────────────
def make_tfidf():
    """글자 n-gram TF-IDF. 한국어는 어미·조사 변화가 많아 단어보다 글자 단위가 안정적이다."""
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)


def make_logreg():
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)


def make_svc():
    return LinearSVC(class_weight="balanced", random_state=SEED)


# ───────────────────────────── 2. 특성 만들기 ─────────────────────────────
def length_features(train, test, use_context):
    """get_xy(use_length=True) 에서 숫자 열(글자 수)만 꺼낸다. 표준화는 cv_splits 가 fold별로 이미 함."""
    X_tr, y_tr = get_xy(train, use_context=use_context, use_length=True)
    X_te, y_te = get_xy(test, use_context=use_context, use_length=True)
    num_cols = [c for c in X_tr.columns if c not in TEXT_COLS]
    return X_tr[num_cols].to_numpy(), y_tr, X_te[num_cols].to_numpy(), y_te


def tfidf_features(train, test, use_context):
    """텍스트 열마다 별도 TF-IDF를 train으로 fit 하고, 열별 결과를 옆으로 붙인다."""
    X_tr, y_tr = get_xy(train, use_context=use_context)
    X_te, y_te = get_xy(test, use_context=use_context)
    tr_parts, te_parts = [], []
    for col in X_tr.columns:  # Comment-only: [Comment] / Context+Comment: [Context, Comment]
        vec = make_tfidf().fit(X_tr[col])
        tr_parts.append(vec.transform(X_tr[col]))
        te_parts.append(vec.transform(X_te[col]))
    return hstack(tr_parts).tocsr(), y_tr, hstack(te_parts).tocsr(), y_te


# ───────────────────────────── 3. 평가 ─────────────────────────────
def score(model, cond, fold, y_true, y_pred):
    rows.append({
        "model": model,
        "condition": cond,
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_hate": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    })


def run_fold(fold, train, test):
    # (1)(2) Dummy: 입력을 보지 않으므로 조건 구분 없이 1회
    _, y_tr = get_xy(train, use_context=False)
    _, y_te = get_xy(test, use_context=False)
    for strategy in ("most_frequent", "stratified"):
        clf = DummyClassifier(strategy=strategy, random_state=SEED)
        clf.fit(np.zeros((len(y_tr), 1)), y_tr)
        score(f"Dummy {strategy}", "-", fold, y_te, clf.predict(np.zeros((len(y_te), 1))))

    for cond, use_context in CONDITIONS.items():
        # (3) 글자 수만 LogReg
        X_tr, y_tr, X_te, y_te = length_features(train, test, use_context)
        score("Length LogReg", cond, fold, y_te, make_logreg().fit(X_tr, y_tr).predict(X_te))

        # (4)(5) TF-IDF + 선형 모델 (같은 특성을 두 모델이 공유)
        X_tr, y_tr, X_te, y_te = tfidf_features(train, test, use_context)
        score("TF-IDF LogReg", cond, fold, y_te, make_logreg().fit(X_tr, y_tr).predict(X_te))
        score("TF-IDF LinearSVC", cond, fold, y_te, make_svc().fit(X_tr, y_tr).predict(X_te))


def mean_std(s):
    """평균 ± 표준편차(표본, ddof=1) 문자열."""
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"


def context_gain(df):
    """모델별 맥락 이득 Δ = (Context+Comment) − (Comment-only). 같은 fold끼리 짝지어 뺀다."""
    paired = df[df["condition"] != "-"].pivot_table(index=["model", "fold"], columns="condition", values=METRICS)
    gain = pd.DataFrame({m: paired[(m, "Context+Comment")] - paired[(m, "Comment-only")] for m in METRICS})
    return gain.groupby(level="model").agg(mean_std)


# ───────────────────────────── 4. 실험로그 (atexit) ─────────────────────────────
def log_row():
    """끝까지 돌았을 때만 실험로그 한 행을 출력한다. 중간에 실패하면 행을 만들지 않는다."""
    if len(rows) != N_EXPECTED or not summary:
        print(f"\n결과 미완 — 로그 생략 (모인 결과 {len(rows)} / {N_EXPECTED})")
        return
    config = f"split=post_group 5-fold StratifiedGroupKFold, seed={SEED}"
    result_file = OUT_CSV.relative_to(ROOT)
    print("\n[실험로그]")
    print("| 실험 이름 | 설정 | 지표 | 결과 파일 | 날짜 |")
    print("|---|---|---|---|---|")
    print(f"| baseline | {config} | {summary['metrics']} | {result_file} | {date.today().isoformat()} |")
    print("\nNotion 기록 제안 (실행하지 않음):")
    print(f'from src.notion_log import log_experiment; log_experiment("baseline", '
          f'{{"split": "post_group 5-fold StratifiedGroupKFold", "seed": {SEED}}}, '
          f'{summary["metrics_dict"]}, "{result_file}")')


# ───────────────────────────── 터미널에서 직접 실행할 때 ─────────────────────────────
if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    atexit.register(log_row)
    np.random.seed(SEED)
    print(f"seed = {SEED}")

    for fold, train, test, _ in cv_splits():
        run_fold(fold, train, test)

    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["model"]))  # 실행 순서대로 표시

    table = df.groupby(["model", "condition"], sort=False)[METRICS].agg(mean_std)
    print("\n[기준선 5-fold 평균 ± 표준편차]")
    print(table.to_string())

    gain = context_gain(df)
    gain = gain.loc[[m for m in order if m in gain.index]]
    print("\n[맥락 이득 Δ = (Context+Comment) − (Comment-only), fold별 짝지어 계산]")
    print(gain.to_string())

    # fold별 원 점수 저장
    if OUT_CSV.exists():
        print(f"\n알림: {OUT_CSV.relative_to(ROOT)} 이(가) 이미 있어 덮어씁니다.")
    df.round(6).to_csv(OUT_CSV, index=False)
    print(f"\n저장: {OUT_CSV.relative_to(ROOT)} ({len(df)}행)")

    # 실험로그용 지표: 조합별 macro-F1 + 모델별 Δmacro-F1
    f1 = df.groupby(["model", "condition"], sort=False)["macro_f1"]
    # 키에 '|' 를 쓰면 마크다운 표 칸이 깨지므로 '/' 로 구분한다
    metrics_dict = {f"macro_f1[{m} / {c}]": round(float(s.mean()), 3) for (m, c), s in f1}
    parts = [f"macro_f1[{m} / {c}]={mean_std(s)}" for (m, c), s in f1]
    parts += [f"Δmacro_f1[{m}]={gain.loc[m, 'macro_f1']}" for m in gain.index]
    summary.update(metrics=", ".join(parts), metrics_dict=metrics_dict)
