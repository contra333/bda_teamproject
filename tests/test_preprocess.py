import contextlib
import hashlib
import io
import unittest

import numpy as np
import pandas as pd

from src import preprocess as pp


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(pp.DATA_PATH)
        cls.saved = pd.read_csv(pp.SPLIT_CSV)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.folds = list(pp.cv_splits_inner(save=False))

    def test_fixed_release_and_row_coverage(self):
        self.assertEqual(
            hashlib.sha256(pp.DATA_PATH.read_bytes()).hexdigest(),
            "d5eb3c7bb4fd97471bd77efd9eb7f98ebc162151ef9725e821510551a4485a38",
        )
        self.assertEqual(self.raw.shape, (3000, 18))
        self.assertEqual(self.raw[pp.TARGET].value_counts().to_dict(), {0: 2236, 1: 764})
        self.assertEqual(pp.make_post_groups(self.raw).nunique(), 77)
        test_ids = pd.concat([test["Index"] for _, _, test, _ in self.folds])
        self.assertEqual(len(test_ids), len(self.raw))
        self.assertTrue(test_ids.is_unique)
        self.assertEqual(set(test_ids), set(self.raw["Index"]))

    def test_no_post_or_context_leakage_and_saved_assignment(self):
        for fold, train, test, inner in self.folds:
            with self.subTest(fold=fold):
                saved = self.saved[self.saved["outer_fold"] == fold]
                self.assertEqual(len(saved), len(self.raw))
                self.assertTrue(saved["Index"].is_unique)
                for role, part in [("train", train), ("test", test)]:
                    self.assertEqual(set(saved.loc[saved["role"] == role, "Index"]), set(part["Index"]))
                for col in ["post_group", "Set", "Context"]:
                    self.assertFalse(set(train[col]) & set(test[col]))
                validation_ids = []
                for k, (itr, iva) in enumerate(inner):
                    fitting, validation = train.iloc[itr], train.iloc[iva]
                    self.assertEqual(len(fitting) + len(validation), len(train))
                    self.assertFalse(set(fitting["Index"]) & set(validation["Index"]))
                    for col in ["post_group", "Context"]:
                        self.assertFalse(set(fitting[col]) & set(validation[col]))
                    self.assertEqual(set(saved.loc[saved["inner_val_fold"] == k, "Index"]), set(validation["Index"]))
                    validation_ids.extend(validation["Index"])
                self.assertEqual(len(validation_ids), len(train))
                self.assertEqual(set(validation_ids), set(train["Index"]))
                self.assertTrue(saved.loc[saved["role"] == "test", "inner_val_fold"].eq(-1).all())

    def test_model_inputs_exclude_annotations_and_preserve_targets(self):
        for _, train, test, _ in self.folds:
            for part in [train, test]:
                for context, length, columns in [
                    (False, False, ["Comment"]),
                    (True, False, ["Context", "Comment"]),
                    (False, True, ["Comment", "comment_len"]),
                    (True, True, ["Context", "Comment", "context_len", "comment_len"]),
                ]:
                    x, y = pp.get_xy(part, use_context=context, use_length=length)
                    self.assertEqual(list(x), columns)
                    expected = self.raw.set_index("Index").loc[part["Index"], pp.TARGET].to_numpy()
                    np.testing.assert_array_equal(y, expected)

    def test_scaling_uses_training_text_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            train, test, scaler = pp.load_and_split()
        raw_train = self.raw.set_index("Index").loc[train["Index"]]
        lengths = raw_train[pp.TEXT_COLS].apply(lambda col: col.str.len())
        np.testing.assert_allclose(scaler.mean_, lengths.mean().to_numpy())
        np.testing.assert_allclose(scaler.scale_, lengths.std(ddof=0).to_numpy())
        np.testing.assert_allclose(train[pp.NUM_COLS].mean(), 0, atol=1e-12)
        raw_test = self.raw.set_index("Index").loc[test["Index"]]
        test_lengths = raw_test[pp.TEXT_COLS].apply(lambda col: col.str.len()).to_numpy()
        np.testing.assert_allclose(test[pp.NUM_COLS], (test_lengths - scaler.mean_) / scaler.scale_)


if __name__ == "__main__":
    unittest.main()
