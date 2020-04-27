from io import StringIO
from unittest import TestCase

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np
from numpy.testing import assert_allclose

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

# Returns an array of global variables and dw that has
# been somewhat fit to the test residues. Useful for
# creating tests in somewhat reasonable parameter spaces.
def _prefit_model():
    return np.array([2.97, 3.52, 13.97, 27574071.42, 97798.68,
                     0.63, 40.79, 30.02, 77.49])

# Returns an array of global variables and dw that has
# been somewhat fit to the test residues. Slight differences
# from _prefit_model to faciliate testing where multiple
# models are needed for input.
def _prefit_model_2():
    return np.array([3.21, 3.69, 13.37, 27574088.88, 97772.34,
                     0.57, 43.21, 32.42, 79.18])

# Returns an array of reference peaks that has been somewhat
# fit to the test residues. 
def _prefit_refpeaks():
    return np.array([124.57 , 126.76, 121.13, 7.76, 8.35, 8.20,
                     3406563.67, 4993584.45, 2168720.03])

class TestOptimizer(TestCase):

    def test_set_bounds(self):
        optimizer = _test_object()
        bounds = ([1, 2, 3], [4, 5, 6])
        optimizer.set_bounds(bounds)
        self.assertEqual(bounds, optimizer.get_bounds())

    def test_ml_optimization_fitness(self):
        params = _prefit_model()

        optimizer = _test_object()
        optimizer.mode = "ml_optimization"

        actual_negLL = optimizer.fitness(params)[0]
        expected_negLL = 82.13685858503707

        self.assertAlmostEqual(actual_negLL, expected_negLL, places=3)

    def test_refpeak_optimization_fitness(self):
        params = _prefit_refpeaks()

        optimizer = _test_object()
        optimizer.l1_model = _prefit_model()
        optimizer.lam = 0.
        optimizer.mode = "reference_optimization"
        optimizer.cs_dist = "rayleigh"

        actual_negLL = optimizer.fitness(params)[0]
        expected_negLL = 86.24555893739519
        self.assertAlmostEqual(actual_negLL, expected_negLL, places=3)

    def test_dw_scale_optimization_fitness(self):
        model = _prefit_model()
        model_with_dw = np.insert(model, 7, .9)
        params = model_with_dw

        optimizer = _test_object()
        optimizer.l1_model = _prefit_model_2()
        optimizer.lam = 0.
        optimizer.mode = "dw_scale_optimization"
        optimizer.cs_dist = "rayleigh"

        # Add noise to each reference peak.
        # When using rayleigh distribution, reference peaks cannot
        # match observed peak because doing so would lead to an
        # inf in the pdf (density) calculation.
        optimizer.reference[["15N_ref", "1H_ref", "I_ref"]] += \
            np.array([[0.1, 0.2, 0.3]]).T

        actual_negLL = optimizer.fitness(params)[0]
        expected_negLL = 84534.24770801092
        self.assertAlmostEqual(actual_negLL, expected_negLL, places=3)

    def test_pfitter(self):
        optimizer = _test_object()
        optimizer.ml_model = _prefit_model()
        optimizer.lam = 0.
        optimizer.mode = "pfitter"
        optimizer.cs_dist = "rayleigh"

        optimizer.reference[["15N_ref", "1H_ref", "I_ref"]] += \
            np.array([[0.1, 0.2, 0.3]]).T

        actual_df = optimizer.fitness()
        # TODO(auberon): Switch to less brittle test once fitness function refactored.
        expected_df = pd.DataFrame({
            '15N': {0: 124.56700000000001, 1: 124.565, 2: 126.77600000000001, 3: 126.777, 4: 121.124, 5: 121.125}, 
            '15N_ref': {0: 124.667, 1: 124.667, 2: 126.97600000000001, 3: 126.97600000000001, 4: 121.42399999999999, 5: 121.42399999999999}, 
            '1H': {0: 7.763, 1: 7.763, 2: 8.349, 3: 8.349, 4: 8.202, 5: 8.201}, 
            '1H_ref': {0: 7.8629999999999995, 1: 7.8629999999999995, 2: 8.549, 3: 8.549, 4: 8.502, 5: 8.502}, 
            'I_ref': {0: 3248421.1, 1: 3248421.1, 2: 5053537.2, 3: 5053537.2, 4: 2062229.3, 5: 2062229.3}, 
            'csfit': {0: 0.0, 1: 0.0005911449393205245, 2: 0.0, 3: 0.00043509098136506013, 4: 0.0, 5: 0.0011225874938422644}, 
            'csp': {0: 0.10198039027185513, 1: 0.10205959043617586, 2: 0.2039607805437108, 3: 0.2039216516214007, 4: 0.3059411708155677, 5: 0.3068827789238098}, 'dw': {0: 40.79, 1: 40.79, 2: 30.02, 3: 30.02, 4: 77.49, 5: 77.49}, 
            'ifit': {0: 3248421.1, 1: 3208792.432251481, 2: 5053537.2, 3: 4959741.54402069, 4: 2062229.3, 5: 2044766.7717852024}, 'intensity': {0: 3248421.0, 1: 3191240.0, 2: 5053537.0, 3: 4964073.0, 4: 2062229.0, 5: 2136456.0}, 
            'residue': {0: 6, 1: 6, 2: 7, 3: 7, 4: 9, 5: 9}, 
            'titrant': {0: 0.0, 1: 7.9, 2: 0.0, 3: 7.9, 4: 0.0, 5: 7.9}, 
            'visible': {0: 150.0, 1: 150.0, 2: 150.0, 3: 150.0, 4: 150.0, 5: 150.0}})

        assert_frame_equal(actual_df, expected_df, check_like=True)

    def test_lfitter(self):
        params = _prefit_model()
        
        optimizer = _test_object()
        optimizer.ml_model = _prefit_model_2()
        optimizer.lam = 0.
        optimizer.mode = "lfitter"
        optimizer.cs_dist = "rayleigh"

        optimizer.reference[["15N_ref", "1H_ref", "I_ref"]] += \
            np.array([[0.1, 0.2, 0.3]]).T

        df = optimizer.fitness(params)
        actual_row = df.loc[[2999]].squeeze()
        # Perform spot-check on a single row.
        # TODO(auberon): Switch to less brittle test once fitness function refactored.
        expected_row = pd.Series([8.69, 150.0, 9, 121.424, 8.502, 2062229.3, 77.49, 2043052, 0.001234],
                                 ['titrant', 'visible', 'residue', '15N_ref', '1H_ref', 'I_ref', 'dw',
       'ifit', 'csfit'])

        assert_series_equal(actual_row, expected_row, check_less_precise=True, check_names=False)

    def test_simulated_peak_generation(self):
        optimizer = _test_object()
        optimizer.ml_model = _prefit_model()
        optimizer.lam = 0.
        optimizer.mode = "simulated_peak_generation"
        optimizer.cs_dist = "rayleigh"

        optimizer.reference[["15N_ref", "1H_ref", "I_ref"]] += \
            np.array([[0.1, 0.2, 0.3]]).T

        actual_df = optimizer.fitness()
        # TODO(auberon): Switch to less brittle test once fitness function refactored.
        expected_df = pd.DataFrame({'15N': {0: 124.56700000000001, 1: 124.565, 2: 126.77600000000001, 3: 126.777, 4: 121.124, 5: 121.125},
                                    '15N_ref': {0: 124.667, 1: 124.667, 2: 126.97600000000001, 3: 126.97600000000001, 4: 121.42399999999999, 5: 121.42399999999999},
                                    '1H': {0: 7.763, 1: 7.763, 2: 8.349, 3: 8.349, 4: 8.202, 5: 8.201},
                                    '1H_ref': {0: 7.8629999999999995, 1: 7.8629999999999995, 2: 8.549, 3: 8.549, 4: 8.502, 5: 8.502},
                                    'I_ref': {0: 3248421.1, 1: 3248421.1, 2: 5053537.2, 3: 5053537.2, 4: 2062229.3, 5: 2062229.3},
                                    'csfit': {0: 0.0, 1: 0.0005911449393205245, 2: 0.0, 3: 0.00043509098136506013, 4: 0.0, 5: 0.0011225874938422644},
                                    'csp': {0: 0.10198039027185513, 1: 0.10205959043617586, 2: 0.2039607805437108, 3: 0.2039216516214007, 4: 0.3059411708155677, 5: 0.3068827789238098}, 'dw': {0: 40.79, 1: 40.79, 2: 30.02, 3: 30.02, 4: 77.49, 5: 77.49},
                                    'ifit': {0: 3248421.1, 1: 3208792.432251481, 2: 5053537.2, 3: 4959741.54402069, 4: 2062229.3, 5: 2044766.7717852024},
                                    'intensity': {0: 3248421.0, 1: 3191240.0, 2: 5053537.0, 3: 4964073.0, 4: 2062229.0, 5: 2136456.0},
                                    'residue': {0: 6, 1: 6, 2: 7, 3: 7, 4: 9, 5: 9},
                                    'titrant': {0: 0.0, 1: 7.9, 2: 0.0, 3: 7.9, 4: 0.0, 5: 7.9},
                                    'visible': {0: 150.0, 1: 150.0, 2: 150.0, 3: 150.0, 4: 150.0, 5: 150.0}})

        assert_frame_equal(actual_df, expected_df, check_like=True)
    
    def test_observed_chemical_shift(self):
        optimizer = _test_object()

        optimizer.nh_scale = 4
        optimizer.larmor = 10

        nitrogen_ref = pd.Series([1, 2, 3])
        nitrogen = pd.Series([6, 5, 4])
        hydrogen_ref = pd.Series([10, 20, 30])
        hydrogen = pd.Series([60, 50, 40])

        actual_shift = optimizer.observed_chemical_shift(nitrogen_ref,
                                                         nitrogen,
                                                         hydrogen_ref,
                                                         hydrogen)
        expected_shift = pd.Series([538.5164807134504, 323.10988842807024, 107.70329614269008])
        assert_series_equal(actual_shift, expected_shift, check_dtype=False)   

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
