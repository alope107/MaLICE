from unittest import TestCase
import pandas as pd
import numpy as np
from io import StringIO

from malice.optimizer import MaliceOptimizer

# Small test dataset.
# Corresponds to data in data/minidev.csv
def _test_residues():
    data = """residue,15N,1H,intensity,titrant,visible
6,124.567,7.763,3248421.0,0.0,150.0
7,126.776,8.349,5053537.0,0.0,150.0
9,121.124,8.202,2062229.0,0.0,150.0
6,124.565,7.763,3191240.0,7.9,150.0
7,126.777,8.349,4964073.0,7.9,150.0
9,121.125,8.201,2136456.0,7.9,150.0"""
    data_file = StringIO(data)
    return pd.read_csv(data_file)

def _test_object():
    return MaliceOptimizer(data=_test_residues(),
                           larmor=500,
                           nh_scale=0.2,
                           lam=0.015)

class TestOptimizer(TestCase):

    def test_set_bounds(self):
        optimizer = _test_object()
        bounds = ([1, 2, 3], [4, 5, 6])
        optimizer.set_bounds(bounds)
        self.assertEqual(bounds,optimizer.get_bounds())

    def test_ml_optimization_fitness(self):
        params = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        optimizer = _test_object()
        optimizer.mode = "ml_optimization"

        actual_negLL = optimizer.fitness(params)[0]
        expected_negLL = 787800164113.927

        self.assertAlmostEqual(actual_negLL, expected_negLL, places=3)

    def test_refpeak_optimization_fitness(self):
        params = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        l1_model = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])

        optimizer = _test_object()
        optimizer.l1_model = l1_model
        optimizer.lam = 0.
        optimizer.mode = "reference_optimization"
        optimizer.cs_dist="rayleigh"

        actual_negLL = optimizer.fitness(params)[0]
        expected_negLL = 203403594105.92865
        self.assertAlmostEqual(actual_negLL, expected_negLL, places=3)
