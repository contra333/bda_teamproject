import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev


class ScoreSumTests(unittest.TestCase):
    def setUp(self):
        outer, inner, assignment = [], [], []
        self.sources = [ev.COMMENT, ev.CONTEXT]
        for fold in range(3):
            for idx in range(1, 13):
                is_test = (idx - 1) // 4 == fold
                inner_fold = (idx // 2) % 2
                assignment.append({
                    "outer_fold": fold, "Index": idx, "role": "test" if is_test else "train",
                    "inner_val_fold": -1 if is_test else inner_fold,
                })
                y = idx % 2
                for seed in (42, 43):
                    for offset, condition in enumerate(self.sources):
                        score = (2 * y - 1) * (0.5 + offset + idx / 10) + (seed - 42) * idx / 6
                        row = {"seed": seed, "fold": fold, "condition": condition, "Index": idx, "y": y}
                        if is_test:
                            outer.append(dict(row, score=score, variant="own", Type=1 if y else 2,
                                              Comment=str(idx)))
                        else:
                            inner.append(dict(row, inner_fold=inner_fold, score=score + fold / 4))
        self.outer = pd.DataFrame(outer)[ev.COLS]
        self.inner = pd.DataFrame(inner)
        self.split = pd.DataFrame(assignment)

    def score_sum(self, outer=None, inner=None):
        p = ev.add_score_sum(self.outer if outer is None else outer,
                             self.inner if inner is None else inner, self.split)
        return p[p["condition"] == ev.SUM].sort_values(["seed", "fold", "Index"])

    def test_reproduces_separate_outer_fold_and_seed_fits(self):
        result = self.score_sum()
        self.assertEqual(len(result), 24)
        for (seed, fold), scored in result.groupby(["seed", "fold"]):
            fit = self.inner.query("seed == @seed and fold == @fold").pivot(
                index=["Index", "y"], columns="condition", values="score").reset_index()
            test = self.outer.query("seed == @seed and fold == @fold").pivot(
                index="Index", columns="condition", values="score")
            model = LogisticRegression(class_weight="balanced").fit(fit[self.sources], fit["y"])
            np.testing.assert_array_equal(scored["Index"], test.index)
            np.testing.assert_allclose(scored["score"], model.decision_function(test[self.sources]), atol=1e-10)

    def test_outer_test_labels_and_other_fold_scores_do_not_affect_fit(self):
        expected = self.score_sum().query("fold == 0")["score"]
        outer = self.outer.copy()
        outer.loc[outer["fold"] == 0, "y"] = 1 - outer.loc[outer["fold"] == 0, "y"]
        outer.loc[outer["fold"] != 0, "score"] *= -100
        inner = self.inner.copy()
        inner.loc[inner["fold"] != 0, "score"] *= -100
        actual = self.score_sum(outer, inner).query("fold == 0")["score"]
        np.testing.assert_array_equal(actual, expected)

    def test_one_outer_fold_can_be_evaluated_on_its_own(self):
        expected = self.score_sum().query("fold == 0")["score"]
        actual = self.score_sum(self.outer.query("fold == 0"), self.inner.query("fold == 0"))
        self.assertEqual(len(actual), len(expected))
        np.testing.assert_array_equal(actual["score"], expected)

    def test_rejects_outer_test_row_in_inner_data(self):
        injected = self.inner.iloc[0].copy()
        injected["Index"] = 1
        invalid = pd.concat([self.inner, injected.to_frame().T], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "분할 기록"):
            self.score_sum(inner=invalid)

    def test_rejects_wrong_inner_fold(self):
        invalid = self.inner.copy()
        invalid.loc[0, "inner_fold"] = 1 - invalid.loc[0, "inner_fold"]
        with self.assertRaisesRegex(ValueError, "분할 기록"):
            self.score_sum(inner=invalid)

    def test_rejects_incomplete_pairs(self):
        with self.assertRaisesRegex(ValueError, "두 조건의 점수"):
            self.score_sum(inner=self.inner.iloc[1:])

    def test_rejects_inconsistent_labels(self):
        invalid = self.inner.copy()
        invalid.loc[0, "y"] = 1 - invalid.loc[0, "y"]
        with self.assertRaisesRegex(ValueError, "분할 기록"):
            self.score_sum(inner=invalid)

    def test_missing_inner_data_does_not_fall_back_to_other_outer_scores(self):
        for inner in (None, self.inner[self.inner["condition"] == ev.COMMENT]):
            with self.subTest(inner_missing=inner is None):
                p = ev.add_score_sum(self.outer, inner, self.split)
                pd.testing.assert_frame_equal(p, self.outer)


if __name__ == "__main__":
    unittest.main()
