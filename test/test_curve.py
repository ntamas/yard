#!/usr/bin/env python

import unittest

from itertools import izip_longest
from textwrap import dedent

from yard.data import BinaryClassifierData
from yard.curve import ROCCurve

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

    def test_auc(self):
        self.assertAlmostEquals(0.95, self.curve.auc(), 8)

    def test_get_points(self):
        expected = [(1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), \
                (0.25, 0.8), (0.0, 0.8), (0.0, 0.6), (0.0, 0.4), \
                (0.0, 0.2), (0.0, 0.0)]
        for obs, exp in zip(self.curve.points, expected):
            for x, y in zip(obs, exp):
                self.assertAlmostEquals(x, y, 5)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    unittest.main(testRunner = runner)

