"""평가: plm.py 가 남긴 행별 점수 파일만 읽어 지표와 판정을 낸다. 학습은 하지 않는다.

읽는 파일
    results/<이름>_preds.csv — 열: model, method, condition, seed, fold, Index, y, score[, variant]
        score = 혐오 logit − 비혐오 logit. 0 이 확률 0.5 에 해당한다.
        variant 열이 없으면 모두 원래 입력("own")으로 본다.

조건
    댓글만 / 글만 / 점수 덧셈 / 따로 읽기 / 같이 읽기 — 파일에 있는 조건만 계산한다.
    점수 덧셈은 댓글만·글만 점수를 LogReg 로 결합한 값이다 (학습 없음). fold k 의 가중치는 나머지 fold 의
    test 점수로만 맞춘다 (fold k 의 정답은 쓰지 않음). 맥락과 댓글을 전혀 섞지 않고 올릴 수 있는 몫을 보여 준다.

임계값 규칙 (혼동행렬·TPR·FPR·macro-F1 은 임계값에 따라 달라진다)
    기본          : score > 0 (확률 0.5). 가설 1·2 발표 수치와 같은 판정이다.
    OOF 조정      : baseline.py 의 "OOF 조정"과 같은 절차. fold 마다 train 내부 validation 점수(plm.py --inner 가 만든
                    <이름>_inner_preds.csv)에서 macro-F1 이 가장 높은 임계값을 고르고, train 전체로 학습한 모델의 test
                    점수에 적용한다. test 는 고르는 데 쓰지 않는다. 내부 점수가 있는 (조건, seed)에만 계산한다.
    임계값 후보는 확률 0.20~0.80 (0.025 간격, baseline.py 의 LogReg 후보와 같음)이고, 동점이면 0.5 에 가까운 값이다.
    AUROC·PR-AUC·짝 안 순서 비율은 임계값과 무관하다.

지표
    TPR(혐오 재현율) = TP / (TP + FN),  FPR = FP / (FP + TN) (특이도 = 1 − FPR),  balanced accuracy = (TPR + 특이도) / 2
    Type별        : Type 1 은 전부 혐오라 TPR, Type 2·3 은 전부 비혐오라 FPR 만 정의된다.
    짝 정답률      : test 에 Type 1·2 가 모두 있는 댓글마다 두 유형의 행을 전부 맞혔는가 (baseline.by_type 과 같은 정의).
    짝 안 순서 비율 : 같은 댓글의 Type 1 행 점수가 Type 2 행 점수보다 높은 비율. 댓글이 같으므로 차이는 글에서만 나온다.
                    댓글만 모델은 0.5 다. 글만 모델도 높게 나올 수 있으므로 함께 읽는 모델이 글만 모델을 넘는지를 본다.
    Type 1·3 macro-F1 / AUROC : 실제 게시글만 (새로 쓴 글인 Type 2 제외).

집계: seed 는 fold 안에서 먼저 평균하고, fold 5개의 평균 ± 표준편차(표본)를 낸다. seed 를 독립 표본으로 세지 않는다.
판정 (baseline.verdict): 평균 Δ 부호 + 5개 중 4개 이상 같은 방향 + 대응 t-검정 p < 0.05.
    사전에 정한 판정 지표는 짝 정답률(1차)·macro-F1·PR-AUC (가설 1 분석에서 쓰던 세 지표)와 혐오 Recall(TPR,
    중간보고서의 보조 지표)이다. 결론 문장은 가설 1 분석과 같이 짝 정답률의 판정으로만 낸다.
    나머지 지표의 비교와, 결과를 본 뒤 추가한 글만 모델과의 비교는 탐색으로 표시한다.

실행
    python src/evaluate.py --preds results/plm_h1_preds.csv          # → results/h1_summary.csv, h1_folds.csv, h1_tests.csv
    results/plm_h1_inner_preds.csv 가 있으면 'OOF 조정' 행이 자동으로 추가된다 (다른 경로는 --inner 로 지정).
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from baseline import TFIDF_MODELS, choose_threshold, verdict
from preprocess import DATA_PATH, ROOT, SPLIT_CSV, TARGET

COMMENT, CONTEXT, SUM, BI, CROSS = "Comment-only", "Context-only", "Score-sum", "Bi-encoder", "Context+Comment"
ORDER = [COMMENT, CONTEXT, SUM, BI, CROSS]
LABEL = {COMMENT: "댓글만", CONTEXT: "글만", SUM: "점수 덧셈 (섞지 않음)", BI: "따로 읽기", CROSS: "같이 읽기"}
# (이름, a, b, 사전에 정한 비교인가). 글만 모델과의 비교는 결과를 본 뒤 추가했으므로 지표와 무관하게 탐색으로 표시한다.
COMPARE = [("맥락 효과 (같이 읽기 − 댓글만)", CROSS, COMMENT, True), ("같이 읽기 − 따로 읽기", CROSS, BI, True),
           ("같이 읽기 − 점수 덧셈", CROSS, SUM, True), ("같이 읽기 − 글만", CROSS, CONTEXT, False),
           ("따로 읽기 − 글만", BI, CONTEXT, False)]
PLANNED = [("pair", "짝 정답률"), ("macro_f1", "macro-F1"), ("pr_auc", "PR-AUC"), ("tpr", "TPR")]  # 사전에 정한 판정 지표
EXPLORE = [("specificity", "특이도"), ("balanced_acc", "balanced acc"), ("auroc", "AUROC"),
           ("pair_order", "짝 안 순서 비율"), ("real_f1", "Type 1·3 macro-F1"), ("auroc_t13", "AUROC(Type 1·3)")]
RULES = ["기본", "OOF 조정"]
GRID, DEFAULT = TFIDF_MODELS["TF-IDF LogReg"][2:4]  # baseline.py 의 LogReg 임계값 후보(확률 0.20~0.80)와 기본값 0.5
KEY = ["model", "method", "condition", "variant", "seed", "fold", "Index"]
COLS = ["condition", "variant", "seed", "fold", "Index", "y", "score", "Type", "Comment"]


# ───────────────────────────── 읽기 ─────────────────────────────
def load(path, model, method="finetune"):
    """행별 점수를 읽고 Type·Comment 를 붙인다. 끊겼다 이어 돌려 겹친 행은 마지막 것만 쓴다."""
    p = pd.read_csv(path)
    if "variant" not in p:
        p["variant"] = "own"
    p = p.drop_duplicates(KEY, keep="last")
    p = p[(p["model"] == model) & (p["method"] == method)].drop(columns=["model", "method"])
    assert len(p), f"{path.name} 에 {model} / {method} 행이 없습니다"
    raw = pd.read_csv(DATA_PATH)[["Index", "Type", "Comment", TARGET]]
    p = p.merge(raw, on="Index", validate="many_to_one")
    assert (p["y"] == p[TARGET]).all(), "저장된 정답과 원본 라벨이 다릅니다"
    warn_if_other_split(p, path.name)
    return p[COLS]


def warn_if_other_split(p, name):
    """행별 점수의 test fold 가 팀의 기준 분할 기록(results/split_assignment.csv)과 같은지 알려 준다.
    다르면 계산은 계속하되, 기준선·다른 가설과 fold 가 맞지 않는다는 점을 표 위에 경고한다."""
    if not SPLIT_CSV.exists():
        return
    rec = pd.read_csv(SPLIT_CSV)
    rec = rec[rec["role"] == "test"].set_index("Index")["outer_fold"]
    same = (p["fold"].to_numpy() == rec.reindex(p["Index"]).to_numpy()).mean()
    if same < 1:
        size = p.drop_duplicates("Index").groupby("fold").size().tolist()
        print(f"⚠ 분할 경고: {name} 의 test fold 가 기준 분할 기록({SPLIT_CSV.name})과 {same:.0%}만 일치합니다.\n"
              f"  이 점수는 다른 분할에서 나왔습니다 (fold별 test 행 수 {size}). 게시글 묶음은 겹치지 않아 누설은 없지만,\n"
              f"  기준선·다른 가설과 같은 fold 가 아니고 내부 validation 점수(--inner)와도 함께 쓸 수 없습니다.\n")
    else:
        print(f"분할 확인: {name} 의 test fold 가 {SPLIT_CSV.name} 기록과 같습니다.\n")


def load_inner(path, model, method="finetune"):
    """train 내부 validation 행별 점수 (plm.py --inner). 4조각이 모두 끝난 (조건, seed, fold)만 쓴다."""
    q = pd.read_csv(path).drop_duplicates(["model", "method", "condition", "seed", "fold", "inner_fold", "Index"], keep="last")
    q = q[(q["model"] == model) & (q["method"] == method)]
    n = q.groupby(["condition", "seed", "fold"])["inner_fold"].transform("nunique")
    return q[n == n.max()] if len(q) else q


def add_score_sum(p):
    """댓글만·글만 점수의 결합. 조건 하나가 없으면 만들지 않는다."""
    own = p[p["variant"] == "own"]  # 바꿔 끼운 입력은 쓰지 않는다
    if not {COMMENT, CONTEXT} <= set(own["condition"]):
        return p
    w = own[own["condition"].isin([COMMENT, CONTEXT])].pivot(
        index=["seed", "fold", "Index", "y", "Type", "Comment"], columns="condition", values="score").dropna().reset_index()
    out = []
    for _, g in w.groupby("seed"):
        for fold in sorted(g["fold"].unique()):
            fit, ap = g[g["fold"] != fold], g[g["fold"] == fold]
            if fit.empty:
                continue
            clf = LogisticRegression(class_weight="balanced").fit(fit[[COMMENT, CONTEXT]], fit["y"])
            out.append(ap.assign(condition=SUM, variant="own", score=clf.decision_function(ap[[COMMENT, CONTEXT]])))
    return pd.concat([p] + [o[COLS] for o in out], ignore_index=True) if out else p


# ───────────────────────────── 지표 ─────────────────────────────
def prob(s):
    """score(혐오 logit − 비혐오 logit)를 혐오 확률로 바꾼다."""
    return 1 / (1 + np.exp(-np.asarray(s, dtype=float)))


def pick_threshold(y, s):
    """내부 validation 점수에서 임계값(확률)을 고른다. baseline.py 의 OOF 조정과 같은 함수·후보·동점 규칙을 쓴다."""
    return choose_threshold(y, prob(s), GRID, DEFAULT)


def pair_accuracy(g, pred):
    """짝 정답률: test 에 Type 1·2 가 모두 있는 댓글마다 두 유형의 행을 전부 맞혔는가."""
    d = g.assign(ok=pred == g["y"].to_numpy())
    d = d[d["Type"].isin([1, 2])]
    both = d.groupby("Comment")["Type"].nunique() == 2
    return d[d["Comment"].isin(both[both].index)].groupby("Comment")["ok"].all().mean()


def pair_order(g):
    """짝 안 순서 비율: 같은 댓글의 Type 1 행 점수가 Type 2 행 점수보다 높은 비율 (동점은 0.5)."""
    vals = []
    for _, c in g[g["Type"].isin([1, 2])].groupby("Comment"):
        s1, s2 = c.loc[c["Type"] == 1, "score"].to_numpy(), c.loc[c["Type"] == 2, "score"].to_numpy()
        if len(s1) and len(s2):
            diff = s1[:, None] - s2[None, :]
            vals.append(((diff > 0) + 0.5 * (diff == 0)).mean())
    return float(np.mean(vals))


def metrics(g, t=DEFAULT):
    """(조건, seed, fold) 하나의 지표. t 는 혐오 확률 기준 임계값이다 (기본 0.5 = score > 0)."""
    y, s, typ = g["y"].to_numpy(), g["score"].to_numpy(), g["Type"].to_numpy()
    pred = (prob(s) > t).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    fp, tn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 0)).sum())
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    real = typ != 2
    return {
        "threshold_prob": t,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "tpr": tpr, "fpr": fpr, "specificity": 1 - fpr,
        "precision_hate": tp / (tp + fp) if tp + fp else 0.0,
        "balanced_acc": (tpr + 1 - fpr) / 2,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "auroc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "pair": pair_accuracy(g, pred),
        "pair_order": pair_order(g),
        "t1_tpr": pred[typ == 1].mean(), "t2_fpr": pred[typ == 2].mean(), "t3_fpr": pred[typ == 3].mean(),
        "real_f1": f1_score(y[real], pred[real], average="macro", zero_division=0),
        "auroc_t13": roc_auc_score(y[real], s[real]),
    }


def fold_metrics(p, inner=None):
    """원래 입력(variant = own)에서 (조건, 규칙, seed, fold)별 지표."""
    own = p[p["variant"] == "own"]
    oof = {k: g for k, g in inner.groupby(["condition", "seed", "fold"])} if inner is not None else {}
    rows = []
    for (cond, seed), g in own.groupby(["condition", "seed"]):
        for fold, test in g.groupby("fold"):
            rows.append({"condition": cond, "rule": RULES[0], "seed": seed, "fold": fold, **metrics(test)})
            if (cond, seed, fold) in oof:
                v = oof[(cond, seed, fold)]
                assert not set(v["Index"]) & set(test["Index"]), "내부 validation 에 test 행이 섞였습니다"
                t = pick_threshold(v["y"].to_numpy(), v["score"].to_numpy())
                rows.append({"condition": cond, "rule": RULES[1], "seed": seed, "fold": fold, **metrics(test, t)})
    return pd.DataFrame(rows)


def variant_table(p):
    """Context 를 바꿔 끼운 입력(variant)별 혐오 판정 비율 — 정답 라벨이 없는 조합이므로 정답률 대신 쓴다.
    이름이 'real_other#0' 처럼 '#번호'로 끝나면 같은 종류의 여러 추출로 보고 행마다 먼저 평균한다."""
    d = p.assign(kind=p["variant"].str.split("#").str[0], hate=(p["score"] > 0).astype(float),
                 p_hate=prob(p["score"]))
    d = d.groupby(["condition", "kind", "seed", "fold", "Type", "Index"])[["hate", "p_hate"]].mean().reset_index()
    f = d.groupby(["condition", "kind", "Type", "fold"])[["hate", "p_hate"]].mean()  # seed·행 평균
    out = f.groupby(level=["condition", "kind", "Type"]).agg(["mean", "std"])
    out.columns = ["혐오 판정 비율", "판정 비율 SD", "평균 혐오 확률", "확률 SD"]
    return out.round(3).reset_index().rename(columns={"condition": "조건", "kind": "붙인 글"})


# ───────────────────────────── 표 ─────────────────────────────
def md(df):
    """표를 발표 자료에 붙이기 쉬운 마크다운으로 만든다."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
    return "\n".join(lines)


def pm(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"


def ci95(d):
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean() - half:+.3f}~{d.mean() + half:+.3f}"


def summarize(folds, conds):
    wide = folds.groupby(["condition", "rule", "fold"]).mean(numeric_only=True).drop(columns="seed")  # seed 평균 먼저
    counts = folds.groupby(["condition", "rule", "seed"])[["TP", "FN", "FP", "TN"]].sum().groupby(level=[0, 1]).mean()
    seeds = folds.groupby(["condition", "rule"])["seed"].unique()
    full = folds["fold"].nunique()
    out = []
    for rule in [r for r in RULES if r in set(folds["rule"])]:
        for c in conds:
            if (c, rule) not in counts.index or len(wide.loc[(c, rule)]) < full:
                continue  # 이 규칙으로 모든 fold 가 끝난 조건만 요약한다 (예: 내부 점수가 일부 fold 에만 있는 경우 제외)
            w, n = wide.loc[(c, rule)], counts.loc[(c, rule)]
            out.append({
                "규칙": rule, "조건": LABEL[c], "seed": " ".join(map(str, sorted(seeds[(c, rule)]))),
                "임계값(확률)": f"{w['threshold_prob'].mean():.2f}",
                "macro-F1": pm(w["macro_f1"]), "TPR": pm(w["tpr"]), "FPR": pm(w["fpr"]),
                "balanced acc": pm(w["balanced_acc"]),
                "TP": f"{n['TP']:.0f}", "FN": f"{n['FN']:.0f}", "FP": f"{n['FP']:.0f}", "TN": f"{n['TN']:.0f}",
                "짝 정답률": f"{w['pair'].mean():.3f}", "Type 1 TPR": f"{w['t1_tpr'].mean():.3f}",
                "Type 2 FPR": f"{w['t2_fpr'].mean():.3f}", "Type 3 FPR": f"{w['t3_fpr'].mean():.3f}",
                "Type 1·3 macro-F1": f"{w['real_f1'].mean():.3f}",
                "AUROC": pm(w["auroc"]), "PR-AUC": pm(w["pr_auc"]), "AUROC(Type 1·3)": pm(w["auroc_t13"]),
                "짝 안 순서 비율": pm(w["pair_order"]),
            })
    return wide, pd.DataFrame(out)


def tfidf_rows(rules, columns):
    """참고 행: 같은 5-fold 의 TF-IDF + LogReg 기준선 (baseline.py 가 저장한 표에서 읽는다, seed 42).
    기준선 표에 없는 지표는 '-' 로 둔다."""
    tf = pd.read_csv(ROOT / "results" / "baseline_folds.csv")
    ty = pd.read_csv(ROOT / "results" / "baseline_by_type.csv")
    out = []
    for rule in rules:
        for cond, name in [(COMMENT, "TF-IDF 댓글만"), (CROSS, "TF-IDF 맥락+댓글")]:
            a = tf[(tf["model"] == "TF-IDF LogReg") & (tf["condition"] == cond) & (tf["rule"] == rule)]
            b = ty[(ty["model"] == "TF-IDF LogReg") & (ty["condition"] == cond) & (ty["rule"] == rule)]
            if a.empty or b.empty:
                continue
            b = b.groupby("group")["accuracy"].mean()
            row = dict.fromkeys(columns, "-")
            row.update({"규칙": rule, "조건": name, "seed": "42", "임계값(확률)": f"{a['threshold'].mean():.2f}",
                        "macro-F1": pm(a["macro_f1"]), "TPR": pm(a["recall_hate"]), "PR-AUC": pm(a["pr_auc"]),
                        "짝 정답률": f"{b['Type 1·2 짝']:.3f}", "Type 1 TPR": f"{b['Type 1']:.3f}",
                        "Type 2 FPR": f"{1 - b['Type 2']:.3f}", "Type 3 FPR": f"{1 - b['Type 3']:.3f}"})
            out.append(row)
    return pd.DataFrame(out, columns=columns)


def conclusion(tests, conds):
    """결론 문장: 1차 지표(짝 정답률)의 판정으로만, 미리 정한 규칙대로 낸다 (가설 1 분석과 같은 규칙)."""
    if not {CROSS, SUM, BI} <= set(conds):
        return None
    v = tests[tests["지표"] == "짝 정답률"].set_index("비교")["판정"]
    sum_v, bi_v = v["같이 읽기 − 점수 덧셈"], v["같이 읽기 − 따로 읽기"]
    if bi_v == "도움 됨":
        return "같이 읽기가 따로 읽기를 넘었다 → 글 구별로 설명되지 않는, 읽는 방식이 만든 효과로 볼 수 있다."
    if bi_v == "해가 됨":
        return "따로 읽기가 같이 읽기를 넘었다 → 입력 단계에서 섞는 것이 도움이 된다는 근거가 없다."
    if sum_v == "도움 됨":
        return ("같이 읽기와 따로 읽기가 구분되지 않는다 → 함께 학습해 판단 단계에서 합치면 충분하고, "
                "토큰 단계에서 섞어야 한다고는 말할 수 없다.")
    return ("같이 읽기가 따로 읽기와도 점수 덧셈과도 구분되지 않는다 → 맥락 효과가 있어도 "
            "'함께 읽어서' 생긴 효과라는 근거는 없다.")


def threshold_effect(folds, conds):
    """임계값을 0.5 에서 OOF 조정으로 바꾸면 지표가 어떻게 변하나. 내부 점수가 있는 같은 (seed, fold)끼리만 비교한다."""
    keys = ["condition", "seed", "fold"]
    m = folds[folds["rule"] == RULES[1]].merge(folds[folds["rule"] == RULES[0]], on=keys, suffixes=("_oof", "_base"))
    m = m.groupby(["condition", "fold"]).mean(numeric_only=True)  # seed 평균 먼저
    out = []
    for c in [c for c in conds if c in m.index.get_level_values(0)]:
        w = m.loc[c]
        row = {"조건": LABEL[c], "fold 수": len(w), "임계값(확률)": f"0.50 → {w['threshold_prob_oof'].mean():.2f}"}
        for k, h in [("macro_f1", "macro-F1"), ("tpr", "TPR"), ("fpr", "FPR")]:
            row[h] = f"{w[k + '_base'].mean():.3f} → {w[k + '_oof'].mean():.3f}"
        out.append(row)
    return pd.DataFrame(out)


def compare(wide, conds):
    tests = []
    for name, a, b, planned in COMPARE:
        if a not in conds or b not in conds:
            continue
        for kind, ms in [("사전 지표", PLANNED), ("탐색", EXPLORE)]:
            for m, h in ms:
                d = (wide.loc[(a, RULES[0])][m] - wide.loc[(b, RULES[0])][m]).to_numpy()
                tests.append({"비교": name, "구분": kind if planned else "탐색", "지표": h, **verdict(d), "95% 구간": ci95(d),
                              "fold별 Δ": " ".join(f"{x:+.3f}" for x in d)})
    return pd.DataFrame(tests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default="results/plm_h1_preds.csv")
    parser.add_argument("--model", default="KcELECTRA-base-v2022")
    parser.add_argument("--method", default="finetune", choices=["finetune", "lora"])
    parser.add_argument("--name", default=None, help="결과 파일 접두사 (기본: plm_h1_preds.csv → h1)")
    parser.add_argument("--inner", default=None, help="train 내부 validation 점수 (기본: <preds 이름>_inner_preds.csv 가 있으면 사용)")
    args = parser.parse_args()
    path = ROOT / args.preds
    name = args.name or path.stem.removeprefix("plm_").removesuffix("_preds")
    out = ROOT / "results"
    inner_path = ROOT / args.inner if args.inner else path.with_name(path.stem.removesuffix("_preds") + "_inner_preds.csv")
    inner = load_inner(inner_path, args.model, args.method) if inner_path.exists() else None
    print(f"내부 validation 점수: {inner_path.name if inner is not None else '없음 → OOF 조정 행은 내지 않는다'}\n")

    p = add_score_sum(load(path, args.model, args.method))
    own = p[p["variant"] == "own"]
    done = own.groupby(["condition", "seed"])["fold"].nunique().unstack()
    print("조건별 끝난 fold 수 (열 = seed)\n", done.to_string(), "\n")
    conds = [c for c in ORDER if c in set(own["condition"])]

    folds = fold_metrics(p, inner=inner)
    folds.round(6).to_csv(out / f"{name}_folds.csv", index=False, encoding="utf-8-sig")
    wide, summary = summarize(folds, conds)
    summary = pd.concat([summary, tfidf_rows(summary["규칙"].unique(), summary.columns)], ignore_index=True)  # 기준선 참고 행
    summary.to_csv(out / f"{name}_summary.csv", index=False, encoding="utf-8-sig")

    a = ["규칙", "조건", "seed", "임계값(확률)", "macro-F1", "TPR", "FPR", "balanced acc", "TP", "FN", "FP", "TN"]
    print("표 1. 분류 지표와 혼동행렬 (혼동행렬은 전체 행 합계의 seed 평균, 나머지는 fold 평균 ± 표준편차)\n" + md(summary[a]) + "\n")
    b = ["규칙", "조건", "짝 정답률", "Type 1 TPR", "Type 2 FPR", "Type 3 FPR", "Type 1·3 macro-F1"]
    print("표 2. Type별 (Type 1 = 혐오, Type 2·3 = 비혐오)\n" + md(summary[b]) + "\n")
    c = ["조건", "AUROC", "PR-AUC", "AUROC(Type 1·3)", "짝 안 순서 비율"]
    print("표 3. 임계값과 무관한 지표\n" + md(summary[summary["규칙"] == RULES[0]][c]) + "\n")
    if RULES[1] in set(folds["rule"]):
        te = threshold_effect(folds, conds)
        te.to_csv(out / f"{name}_threshold.csv", index=False, encoding="utf-8-sig")
        print("표 3-1. 임계값 0.5 → OOF 조정 (내부 점수가 있는 같은 seed·fold 끼리, fold 평균)\n" + md(te) + "\n")

    tests = compare(wide, conds)
    if len(tests):
        tests.to_csv(out / f"{name}_tests.csv", index=False, encoding="utf-8-sig")
        print("표 4. 판정 (기본 임계값). '탐색' 행은 사후에 본 지표이므로 경계선의 p 값을 결론으로 쓰지 않는다\n" + md(tests) + "\n")
        msg = conclusion(tests, conds)
        if msg:
            print("결론 (1차 지표 = 짝 정답률):", msg, "\n")

    if set(p["variant"]) - {"own"}:
        vt = variant_table(p)
        vt.to_csv(out / f"{name}_variants.csv", index=False, encoding="utf-8-sig")
        print("표 5. 붙인 글(variant)별 혐오 판정 비율 (fold 평균과 표준편차)\n" + md(vt) + "\n")

    print("저장: " + ", ".join(f.name for f in sorted(out.glob(f"{name}_*.csv"))))
