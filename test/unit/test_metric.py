import unittest
from test.unit.test_helpers import TensorTestCase

from joeynmt.metrics import chrf, bleu, token_accuracy


class TestMetrics(TensorTestCase):

    def test_chrf_without_whitespace(self):
        hyp1 = ["t est"]
        ref1 = ["tez t"]
        score1 = chrf(hyp1, ref1, remove_whitespace=True)
        hyp2 = ["test"]
        ref2 = ["tezt"]
        score2 = chrf(hyp2, ref2, remove_whitespace=True)
        self.assertAlmostEqual(score1, score2)
        self.assertAlmostEqual(score1, 0.271, places=3)

    def test_chrf_with_whitespace(self):
        hyp = ["これはテストです。"]
        ref = ["これは テストです。"]
        score = chrf(hyp, ref, remove_whitespace=False)
        self.assertAlmostEqual(score, 0.558, places=3)

    def test_bleu_13a(self):
        hyp = ["this is a test."]
        ref = ["this is a tezt."]
        score = bleu(hyp, ref, tokenize="13a")
        self.assertAlmostEqual(score, 42.729, places=3)

    def test_bleu_ja_mecab(self):
        try:
            hyp = ["これはテストです。"]
            ref = ["あれがテストです。"]
            score = bleu(hyp, ref, tokenize="ja-mecab")
            self.assertAlmostEqual(score, 39.764, places=3)
        except Exception as e:
            raise unittest.SkipTest(f"{e} Skip.")

    def test_token_acc_level_char(self):
        # if len(hyp) > len(ref)
        hyp = [list("tests")]
        ref = [list("tezt")]
        #level = "char"
        score = token_accuracy(hyp, ref)
        self.assertEqual(score, 60.0)

        # if len(hyp) < len(ref)
        hyp = [list("test")]
        ref = [list("tezts")]
        #level = "char"
        score = token_accuracy(hyp, ref)
        self.assertEqual(score, 75.0)
