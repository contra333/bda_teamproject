import unittest
import pandas as pd
from src.preprocess import load_data, fixed_assignment, check_assignment, validate, model_inputs


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.assignment = fixed_assignment(cls.data)

    def test_release_and_reproducibility(self):
        self.assertEqual(self.data.shape, (3000,18))
        self.assertEqual(self.data['Hate speech'].value_counts().to_dict(), {0:2236,1:764})
        pd.testing.assert_frame_equal(self.assignment, fixed_assignment(self.data))
        self.assertEqual(self.assignment['split'].value_counts().to_dict(), {'train':1770,'valid':639,'test':591})
        self.assertEqual(self.assignment.groupby('Set')['split'].nunique().max(),1)
        self.assertEqual(self.assignment.Index.nunique(),3000)

    def test_reject_blank_text_invalid_labels_and_duplicate_ids(self):
        for column,value in [('Comment',' '),('Context',None),('Hate speech',2),('Set',None)]:
            bad=self.data.copy();bad.loc[0,column]=value
            with self.subTest(column=column),self.assertRaises(ValueError):validate(bad)
        bad=self.data.copy();bad.loc[0,'Index']=bad.loc[1,'Index']
        with self.assertRaises(ValueError):validate(bad)

    def test_metadata_nulls_are_not_removed(self):
        validate(self.data)
        self.assertEqual(self.data.loc[self.data.Type.eq(2),'Date'].isna().sum(),764)

    def test_detect_cross_split_post(self):
        bad=self.assignment.copy()
        group=bad[bad.Set.eq(bad.Set.iloc[0])].index
        bad.loc[group[0],'split']='test' if bad.loc[group[1],'split']!='test' else 'train'
        with self.assertRaises(ValueError):check_assignment(self.data,bad)

    def test_text_only_inputs_share_targets(self):
        a,y=model_inputs(self.data,'comment_only')
        b,z=model_inputs(self.data,'context_comment')
        self.assertEqual(list(a),['Comment']);self.assertEqual(list(b),['Context','Comment'])
        pd.testing.assert_series_equal(y,z)
        pd.testing.assert_series_equal(a.Comment,b.Comment)
        with self.assertRaises(ValueError):model_inputs(self.data,'metadata')


if __name__=='__main__':unittest.main()
