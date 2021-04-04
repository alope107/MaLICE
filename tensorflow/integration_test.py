from numpy.testing import assert_almost_equal
from unittest import TestCase
from complex_tensorflow import run_malice, parse_args
from malice.seeds import set_base_seed


# Hackish test while doing initial refactor
class IntegrationTest(TestCase):
    def test_end_to_end(self):
        eKd_exp, ekoff_exp, edR2, eamp_scaler, edeltaw_array = [3.9810380935668945, 1.4728188514709473, 0.5954989790916443, 18113388.0, [13.522107, 2.8350213, 1.6546993, 7.6595535, 4.849515, 5.212343, 7.1682625, 7.575306, 4.1894393, 6.526525, 9.074759, 9.550445, 7.2962356, 7.1913233, 10.387511, 7.4521394, 7.46889, 7.617695, 6.9716573, 15.397915, 16.457619, 5.7181535, 5.5802608, 7.995584, 8.328926, 0.011092223, 4.8779144, 7.7701764, 7.5603943, 4.970913, 8.434852, 7.624908, 8.029148, 7.616623, 6.0354366, 6.6349297, 10.521122, 11.348248, 7.17915, 11.498405, 22.519697, 20.191044, 18.130934, 13.494603, 26.375412, 16.698559, 18.2797, 16.488384, 16.374643, 10.329437, 9.626149, 11.356569, 13.448503, 7.6704845, 6.474053, 10.989623, 7.842528, 6.085612, 4.501616, 5.205662, 7.116442, 4.43272, 0.07158047, 6.7557373, 6.174183, 6.4427805, 7.5173554, 6.24484, 5.838639, 5.6753564, 7.715204, 5.419614, 6.496674, 16.782793, 10.080117, 16.516317, 27.472027, 17.350548, 3.7314985, 18.813816, 11.665882, 10.407312, 9.249641, 14.838382, 16.787249, 13.778171, 9.168715, 10.224032, 6.7525873, 14.612611, 10.218408, 15.815498, 12.858357, 8.025757, 6.5960417, 7.046151, 5.942494, 4.674952, 6.527978, 6.2858143, 6.0835752, 14.654378, 25.755089, 4.788774]]
        args = parse_args(["~/programming/malice/MaLICE/complex/data/verl_200111.csv", "--seed",  "1337"])
        set_base_seed(args.seed)
        aKd_exp, akoff_exp, adR2, aamp_scaler, adeltaw_array = run_malice(args)
        assert_almost_equal(eKd_exp, aKd_exp)
        assert_almost_equal(ekoff_exp, akoff_exp)
        assert_almost_equal(edR2, adR2)
        assert_almost_equal(eamp_scaler, aamp_scaler)
        assert_almost_equal(edeltaw_array, adeltaw_array, decimal=4)