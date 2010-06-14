#!/usr/bin/env python

import unittest

from itertools import izip_longest
from textwrap import dedent

from yard.data import BinaryConfusionMatrix, BinaryClassifierData

class BinaryConfusionMatrixTest(unittest.TestCase):
    def setUp(self):
        self.matrices = [\
            BinaryConfusionMatrix(tp=63, fp=28, fn=37, tn=72),
            BinaryConfusionMatrix(tp=77, fp=77, fn=23, tn=23),
            BinaryConfusionMatrix(tp=24, fp=88, fn=76, tn=12),
            BinaryConfusionMatrix(data=[[88, 24], [12, 76]]),
            BinaryConfusionMatrix(tp=100, fp=0, fn=0, tn=100)
        ]

    def test_fdn(self):
        expected = [0.545, 0.23, 0.44, 0.56, 0.5]
        for matrix, exp in zip(self.matrices, expected):
            self.assertAlmostEqual(matrix.fdn(), exp, 2)

    def test_fdp(self):
        expected = [0.455, 0.77, 0.56, 0.44, 0.5]
        for matrix, exp in zip(self.matrices, expected):
            self.assertAlmostEqual(matrix.fdp(), exp, 2)

    def test_tpr(self):
        expected = [0.63, 0.77, 0.24, 0.76, 1.0]
        for matrix, exp in zip(self.matrices, expected):
            self.assertAlmostEqual(matrix.tpr(), exp, 2)

    def test_fpr(self):
        expected = [0.28, 0.77, 0.88, 0.12, 0.0]
        for matrix, exp in zip(self.matrices, expected):
            self.assertAlmostEqual(matrix.fpr(), exp, 2)

    def test_fdr(self):
        expected = [0.30769, 0.5, 0.78571, 0.13636, 0.0]
        for matrix, exp in zip(self.matrices, expected):
            self.assertAlmostEqual(matrix.fdr(), exp, 2)

    def test_accuracy(self):
        expected = [0.675, 0.50, 0.18, 0.82, 1.0]
        for matrix, exp in zip(self.matrices, expected):
            self.assertAlmostEqual(matrix.accuracy(), exp, 2)

    def test_odds_ratio(self):
        expected = [4.378, 1.0, 0.043, 23.222]
        for matrix, exp in zip(self.matrices, expected):
            self.assertAlmostEqual(matrix.odds_ratio(), exp, 2)
        self.assertEquals(str(self.matrices[4].odds_ratio()), "inf")


class BinaryClassifierDataTest(unittest.TestCase):
    def setUp(self):
        self.data = BinaryClassifierData([\
            (0.1, 0),
            (0.2, 0),
            (0.3, 0),
            (0.4, 1),
            (0.5, 0),
            (0.6, 1),
            (0.7, 1),
            (0.8, 1),
            (0.9, 1)
        ])

    def test_get_confusion_matrix(self):
        mat = self.data.get_confusion_matrix(0.2)
        self.assertEquals(repr(mat), "BinaryConfusionMatrix(tp=5, fp=3, fn=0, tn=1)")
        mat = self.data.get_confusion_matrix(0.5)
        self.assertEquals(repr(mat), "BinaryConfusionMatrix(tp=4, fp=1, fn=1, tn=3)")
        mat = self.data.get_confusion_matrix(0.75)
        self.assertEquals(repr(mat), "BinaryConfusionMatrix(tp=2, fp=0, fn=3, tn=4)")
        mat = self.data.get_confusion_matrix(1.0)
        self.assertEquals(repr(mat), "BinaryConfusionMatrix(tp=0, fp=0, fn=5, tn=4)")

    def test_iter_confusion_matrices(self):
        expected = """\
        tp=5, fp=4, fn=0, tn=0
        tp=5, fp=3, fn=0, tn=1
        tp=5, fp=2, fn=0, tn=2
        tp=5, fp=1, fn=0, tn=3
        tp=4, fp=1, fn=1, tn=3
        tp=4, fp=0, fn=1, tn=4
        tp=3, fp=0, fn=2, tn=4
        tp=2, fp=0, fn=3, tn=4
        tp=1, fp=0, fn=4, tn=4
        tp=0, fp=0, fn=5, tn=4"""
        expected = ["BinaryConfusionMatrix(%s)" % line \
                    for line in dedent(expected).split("\n")]
        for (threshold, matrix), expected in \
                izip_longest(self.data.iter_confusion_matrices(), expected):
            self.assertEquals(repr(matrix), expected)
            self.assertEquals(matrix, self.data.get_confusion_matrix(threshold))

        expected = """\
        tp=5, fp=4, fn=0, tn=0
        tp=5, fp=2, fn=0, tn=2
        tp=4, fp=1, fn=1, tn=3
        tp=2, fp=0, fn=3, tn=4
        tp=0, fp=0, fn=5, tn=4"""
        expected = ["BinaryConfusionMatrix(%s)" % line \
                    for line in dedent(expected).split("\n")]
        for (threshold, matrix), expected in \
                izip_longest(self.data.iter_confusion_matrices(4), expected):
            self.assertEquals(repr(matrix), expected)
            self.assertEquals(matrix, self.data.get_confusion_matrix(threshold))


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    unittest.main(testRunner = runner)

