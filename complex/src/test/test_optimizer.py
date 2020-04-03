from unittest import TestCase
import pandas as pd

from malice.optimizer import MaliceOptimizer

def _test_residues():
    return pd.DataFrame( {
        "residue" : [],
        "15N" : [],
        "1H" : [],
        "intensity" : [],
        "titrant" : [],
        "visible" : []
    })

def _test_object():
    return MaliceOptimizer(data=_test_residues())

class TestOptimizer(TestCase):

    def test_set_bounds(self):
        optimizer = _test_object()
        bounds = ([1, 2, 3], [4, 5, 6])
        optimizer.set_bounds(bounds)
        bad_bounds = ([1, 2, 3], [4, 5, 7])
        self.assertEqual(bad_bounds,optimizer.get_bounds())