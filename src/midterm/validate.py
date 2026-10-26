"""제공 결과와 재현 결과, 분할 및 댓글 기준 상한을 점검한다."""
SEED = 42

import contextlib
import hashlib
import io
import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from .preprocess import ROOT, DATA_PATH, make_post_groups
from .preprocess_v2 import cv_splits_v2


def comment_oracle(frame):
    """각 고유 댓글에 한 라벨을 주는 모든 규칙의 macro-F1 최댓값.

    test 정답을 사용하는 사후 분석이며 학습/예측 모델이 아니다.
    TP 개수별 최소 FP를 동적 계획법으로 구한다. 음성만 있는 댓글은
    항상 음성으로 두는 것이 최적이다. TP가 같으면 FP가 작을수록
    양쪽 클래스의 F1이 증가하므로 이 상태 축약이 정확하다.
    """
    grouped = frame.groupby("Comment")["Hate speech"].agg(["sum", "count"])
    positive = int(frame["Hate speech"].sum())
    negative = len(frame) - positive
    assert positive and negative
    min_fp = np.full(positive + 1, len(frame) + 1, dtype=int)
    min_fp[0] = 0
    for row in grouped.itertuples():
        p, n = int(row.sum), int(row.count - row.sum)
        if p:
            min_fp[p:] = np.minimum(min_fp[p:], min_fp[:-p].copy() + n)
    tp = np.arange(positive + 1)
    feasible = min_fp <= negative
    tp, fp = tp[feasible], min_fp[feasible]
    fn, tn = positive - tp, negative - fp
    scores = tp / (2 * tp + fp + fn) + tn / (2 * tn + fp + fn)
    k = int(scores.argmax())
    return {"macro_f1": float(scores[k]), "tp": int(tp[k]), "fp": int(fp[k])}


def check_splits():
    raw = pd.read_csv(DATA_PATH)
    group = make_post_groups(raw)
    assert len(raw) == 3000 and group.nunique() == 77
    assert raw[["Context", "Comment", "Hate speech"]].notna().all().all()
    reference = pd.read_csv(ROOT / "results/midterm/provided/split_assignment_v2.csv")
    coverage = []
    with contextlib.redirect_stdout(io.StringIO()):
        folds = list(cv_splits_v2(save=False))
    for fold, train, test, inner in folds:
        ref = reference[reference.outer_fold == fold]
        assert set(test.Index) == set(ref.loc[ref.role == "test", "Index"])
        assert set(train.Index) == set(ref.loc[ref.role == "train", "Index"])
        assert not set(train.post_group) & set(test.post_group)
        assert not set(train.Context) & set(test.Context)
        coverage.extend(test.Index)
        seen = []
        for k, (itr, iva) in enumerate(inner):
            a, b = train.iloc[itr], train.iloc[iva]
            assert not set(a.post_group) & set(b.post_group)
            assert not set(a.Context) & set(b.Context)
            assert set(b.Index) == set(ref.loc[(ref.role == "train") & (ref.inner_val_fold == k), "Index"])
            seen.extend(iva)
        assert sorted(seen) == list(range(len(train)))
    assert sorted(coverage) == sorted(raw.Index)
    return raw, folds


def main():
    dest = ROOT / "results/midterm"
    raw, folds = check_splits()
    manifest = json.loads((dest / "source_manifest.json").read_text())
    assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == manifest["raw_sha256"]
    checks = []
    for name in manifest["provided_files"]:
        provided = dest / "provided" / name
        assert hashlib.sha256(provided.read_bytes()).hexdigest() == manifest["provided_files"][name]
        reproduced = dest / "reproduced" / name
        if not reproduced.exists():
            assert name == "plm_folds.csv", name
            checks.append({"file": name, "status": "provided_only", "rows": len(pd.read_csv(provided))})
            continue
        a, b = pd.read_csv(provided), pd.read_csv(reproduced)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-6, rtol=0)
        checks.append({"file": name, "status": "match_atol_1e-6", "rows": len(a)})
    oracle = []
    for fold, _, test, _ in folds:
        oracle.append({"fold": fold, **comment_oracle(raw[raw.Index.isin(test.Index)])})
    pd.DataFrame(oracle).to_csv(dest / "comment_oracle_check.csv", index=False)
    report = {"source_data_sha256": manifest["raw_sha256"],
              "environment": {"python": platform.python_version(), **{k: version(k) for k in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]}},
              "splits": {"rows": 3000, "post_groups": 77, "outer_folds": 5, "inner_folds": 4, "test_once_per_row": True, "group_overlap": 0, "context_overlap": 0},
              "csv_comparisons": checks, "comment_oracle_mean": float(np.mean([x["macro_f1"] for x in oracle])),
              "scope": "CPU baseline reproduction; PLM outputs supplied by team, no PLM retraining"}
    (dest / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
