from numpy.testing import assert_almost_equal
from unittest import TestCase
from complex_tensorflow import run_malice, parse_args
from malice.seeds import set_base_seed


# Hackish test while doing initial refactor
class IntegrationTest(TestCase):
    def test_end_to_end(self):
        eKd_exp, ekoff_exp, edR2, eamp_scaler, edeltaw_array = [3.981060266494751, 1.4720667600631714, 0.5956617593765259, 18113388.0, [13.515818, 2.8336287, 1.6535664, 6.7334986, 4.8465905, 5.2088447, 7.1517963, 7.572007, 4.185726, 6.523493, 9.0690365, 9.546257, 7.292441, 7.187559, 10.382393, 7.448577, 7.4653196, 7.612925, 6.967941, 15.391258, 16.449902, 5.9173536, 5.5763803, 7.990082, 8.322783, 0.011045864, 4.8749614, 7.7661166, 7.555913, 4.9678707, 8.430879, 7.6205397, 8.025847, 7.6123877, 6.032261, 6.631436, 10.514952, 11.341634, 7.174754, 11.493471, 22.51336, 20.185894, 18.126547, 13.488669, 26.370153, 16.692478, 18.271591, 16.481752, 16.369135, 10.325078, 5.657492, 11.350806, 13.4423685, 7.666963, 6.4693346, 10.983212, 5.6617007, 6.082653, 4.499747, 5.2025495, 7.1123614, 4.43002, 0.071481876, 6.751646, 6.1714034, 6.4404287, 7.54365, 6.2411065, 5.8363338, 8.249662, 7.711096, 5.41617, 6.4933205, 16.778519, 10.073975, 16.510084, 27.467592, 17.34245, 7.542745, 18.80886, 11.659721, 10.401185, 9.246108, 14.83292, 16.779665, 13.773049, 9.163867, 10.219204, 6.7483644, 14.606525, 10.214202, 15.807807, 12.850472, 6.946363, 6.5926127, 7.043268, 5.93914, 4.6727786, 6.5249753, 6.281967, 6.080351, 14.6481285, 25.748365, 4.7853203]]
        args = parse_args(["~/programming/malice/MaLICE/complex/data/verl_200111.csv", "--seed",  "1337"])
        set_base_seed(args.seed)
        aKd_exp, akoff_exp, adR2, aamp_scaler, adeltaw_array = run_malice(args)
        assert_almost_equal(eKd_exp, aKd_exp)
        assert_almost_equal(ekoff_exp, akoff_exp)
        assert_almost_equal(edR2, adR2)
        assert_almost_equal(eamp_scaler, aamp_scaler)
        assert_almost_equal(edeltaw_array, adeltaw_array, decimal=4)