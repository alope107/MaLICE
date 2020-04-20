from io import StringIO
from unittest import TestCase

import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np

from malice.optimizer import MaliceOptimizer, regularization_penalty

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

    def test_regularization_penalty(self):
        vector = np.array([5, -6, 10])
        lam = .5

        actual_penalty = regularization_penalty(lam, vector)
        expected_penalty = 10.5

        self.assertAlmostEqual(actual_penalty, expected_penalty)

    def test_compute_fits(self):
        optimizer = _test_object()

        # TODO(auberon): Refactor so this logic doesn't need to be in the test
        residue_params = optimizer.reference.copy()
        residue_params['dw'] = np.array([7, 8, 9])
        df = pd.merge(optimizer.data, residue_params, on='residue')

        actual_cshat, actual_ihat = optimizer.compute_fits(1, 2, 3, 4, df)
        expected_cshat = pd.Series([0., 0.343206, 0., 0.391781, 0., 0.440175])
        expected_ihat = pd.Series([3.248421e+06, 2.361621e+01, 5.053537e+06, 
                                   2.272775e+01, 2.062229e+06, 2.180010e+01])

        assert_series_equal(actual_cshat, expected_cshat, check_dtype=False)
        assert_series_equal(actual_ihat, expected_ihat, check_dtype=False)
