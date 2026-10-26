"""기존 W5 분할과 독립적인 중간발표 분할·계산 검증."""
import itertools
import unittest

import pandas as pd
from sklearn.metrics import f1_score

from src.midterm.validate import check_splits, comment_oracle
from src.midterm.preprocess import get_xy


class MidtermChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.folds = check_splits()

    def test_all_reference_assignments_and_group_boundaries(self):
        self.assertEqual(len(self.folds), 5)

    def test_model_inputs_exclude_identifiers_and_labels(self):
        for _, train, test, _ in self.folds:
            for frame in [train, test]:
                for context in [False, True]:
                    x, y = get_xy(frame, use_context=context)
                    self.assertEqual(list(x), ["Context", "Comment"] if context else ["Comment"])
                    self.assertEqual(len(x), len(y))

    def test_comment_oracle_matches_exhaustive_search(self):
        # 한 양성 댓글에 음성이 여러 개 붙으면 max(label) 규칙은 최적이 아니다.
        frame = pd.DataFrame({"Comment": ["a"]*5+["b"]*2+["c"]*2+["d"],
                              "Hate speech": [1,0,0,0,0,1,0,1,1,0]})
        values = frame.Comment.unique()
        exhaustive = max(f1_score(frame["Hate speech"], frame.Comment.map(dict(zip(values, labels))), average="macro")
                         for labels in itertools.product([0,1], repeat=len(values)))
        self.assertAlmostEqual(comment_oracle(frame)["macro_f1"], exhaustive)


if __name__ == "__main__":
    unittest.main()
