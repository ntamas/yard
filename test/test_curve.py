#!/usr/bin/env python

import unittest

from itertools import izip_longest
from textwrap import dedent

from yard.data import BinaryClassifierData
from yard.curve import Curve, ROCCurve


class CurveTest(unittest.TestCase):
    def setUp(self):
        self.data = Curve([(1,2), (3,4), (5,5), (7,0)])

    def test_get_interpolated_point(self):
        self.failUnless(self.data.get_interpolated_point(1)   ==   (1, 2))
        self.failUnless(self.data.get_interpolated_point(5)   ==   (5, 5))
        self.failUnless(self.data.get_interpolated_point(2)   ==   (2, 3))
        self.failUnless(self.data.get_interpolated_point(2.5) == (2.5, 3.5))
        self.failUnless(self.data.get_interpolated_point(0)   ==   (0, 1))
        self.failUnless(self.data.get_interpolated_point(-1)  ==  (-1, 0))
        self.failUnless(self.data.get_interpolated_point(9)   ==  (9, -5))

    def test_resample(self):
        self.data.resample([0, 2, 4, 6, 8])
        self.failUnless(self.data.points == \
             [(0, 1), (2, 3), (4, 4.5), (6, 2.5), (8, -2.5)]
        )

class ROCCurveTest(unittest.TestCase):
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
        self.curve = ROCCurve(self.data)

    def test_auc_superclass(self):
        self.assertAlmostEquals(0.95, Curve.auc(self.curve), 8)

    def test_auc(self):
        self.assertAlmostEquals(0.95, self.curve.auc(), 8)

    def test_get_points(self):
        expected = reversed([(1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), \
                (0.25, 0.8), (0.0, 0.8), (0.0, 0.6), (0.0, 0.4), \
                (0.0, 0.2), (0.0, 0.0)])
        for obs, exp in zip(self.curve.points, expected):
            for x, y in zip(obs, exp):
                self.assertAlmostEquals(x, y, 5)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    unittest.main(testRunner = runner)

