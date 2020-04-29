from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as stats

ComplexGlobals = namedtuple("ComplexGlobals", ["kd_exp", 
                                               "koff_exp",
                                               "dr2",
                                               "amp_scaler",
                                               "i_noise",
                                               "cs_noise",
                                               "scale"])

ComplexSolution = namedtuple("ComplexSolution", ["global_params", "df"])

# Returns L1 norm of a vector scaled by the given factor.
def regularization_penalty(lam, vector):
    return lam * np.linalg.norm(vector, 1)

def merged_residue_df(observed_df, reference_df):
    return pd.merge(observed_df, reference_df, on='residue')

class ComplexProblem(ABC):

    def __init__(self, data=None, larmor=500, lam=0.0, cs_dist='gaussian',
                 nh_scale=0.2, bounds=None, base_reference=None, base_globals=None):
        self.data = data
        self.larmor = larmor
        self.lam = lam
        self.nh_scale = nh_scale
        self.bounds = bounds
        self.cs_dist = cs_dist
        self.base_reference = self.get_base_reference(reference)
        self.base_globals = base_globals

    def get_base_reference(self, reference):
        if reference is not None:
            return reference.copy()

        reference = pd.DataFrame()
        residues = list(self.data.groupby('residue').groups.keys())
        for res in residues:
            resdata = self.data.copy()[self.data.residue == res]
            row = resdata.loc[resdata.titrant == np.min(resdata.titrant),
                              ['residue', '15N', '1H', 'intensity']].mean(axis=0)
            reference = reference.append(row, ignore_index=True)
        reference = reference.rename(columns={'intensity': 'I_ref',
                                              '15N': '15N_ref',
                                              '1H': '1H_ref'})
        return reference

    def set_bounds(self, bds):
        self.bounds = bds

    def get_bounds(self):
        return self.bounds

    def get_scipy_bounds(self):
        return tuple([(self.bounds[0][x], self.bounds[1][x]) for x in range(len(self.bounds[0]))])

     # Returns fits for chemical shift and intensity!
    def compute_fits(self, kd_exp, koff_exp, dr2, amp_scaler, df):
        Kd = np.power(10, kd_exp)
        koff = np.power(10, koff_exp)
        kon = koff/Kd

        dimer = ((df.visible + df.titrant + Kd) -
                 np.sqrt(np.power((df.visible + df.titrant + Kd), 2) -
                 4*df.visible * df.titrant))/2
        pb = dimer/df.visible
        pa = 1 - pb

        # Assume 1:1 stoichiometry for now
        free_titrant = df.titrant - dimer
        kr = koff
        kf = free_titrant * kon
        kex = kr + kf

        # Adjust the amplitude if the visible concentration is not equal
        # for all points
        visibleconc = list(self.data.visible.drop_duplicates())
        if len(visibleconc) > 1:
            amp_scaler = amp_scaler*(self.data.visible/np.mean(visibleconc))

        broad_denom = np.square(np.square(kex) + (1-5*pa*pb)*np.square(df.dw)) + 4*pa*pb*(1-4*pa*pb)*np.power(df.dw, 4)

        # Compute the fits
        i_broad = pa*pb*np.square(df.dw)*kex * (np.square(kex)+(1-5*pa*pb)*np.square(df.dw))/broad_denom
        ihat = df.I_ref/(pa + pb + df.I_ref*(pb*dr2 + i_broad)/amp_scaler)
        cs_broad = pa*pb*(pa-pb)*np.power(df.dw, 3) * (np.square(kex)+(1-3*pa*pb)*np.square(df.dw))/broad_denom
        cshat = pb*df.dw - cs_broad

        return cshat, ihat

    def observed_chemical_shift(self, nitrogen_ref, nitrogen,
                                hydrogen_ref, hydrogen):
        n_diff = nitrogen - nitrogen_ref
        h_diff = hydrogen - hydrogen_ref
        n_diff_scaled = self.nh_scale * n_diff
        shift = np.sqrt(np.square(n_diff_scaled) + np.square(h_diff))
        return self.larmor * shift

    def fitness(self, params=None):
        global_params = self.get_global_params(params)
        residue_params = self.get_residue_params(params)

        residue_params['dw'] *= global_params.scale

        df = merged_residue_df(self.data, residue_params)

        cshat, ihat = self.compute_fits(global_params.kd_exp,
                                        global_params.koff_exp,
                                        global_params.dr2,
                                        global_params.amp_scaler, 
                                        df)
        
        csobs = self.observed_chemical_shift(df['15N_ref'], df['15N'], df['1H_ref'], df['1H'])

        # Compute the likelihoods
        logLL_int = np.sum(stats.norm.logpdf(df.intensity, loc=ihat, scale=global_params.i_noise))

        if self.cs_dist == 'gaussian':
            logLL_cs = np.sum(stats.norm.logpdf(csobs, loc=cshat, scale=global_params.cs_noise))
        elif self.cs_dist == 'rayleigh':
            logLL_cs = np.sum(stats.rayleigh.logpdf(np.abs(csobs-cshat)/global_params.cs_noise) - np.log(global_params.cs_noise))
        else:
            print('INVALID CS DISTRIBUTION')
            return 0

        negLL = -1*(logLL_int + logLL_cs - regularization_penalty(self.lam, df.dw))

        return(negLL, )

    @abstractmethod
    def get_global_params(self, params):
        pass

    @abstractmethod
    def get_residue_params(self, params):
        pass


# "ml_optimization"
# Optimizes globals + dw
class MixedProblem(ComplexProblem):
    def get_global_params(self, params):
        return ComplexGlobals(kd_exp=params[0],
                              koff_exp=params[1],
                              dr2=params[2],
                              amp_scaler=params[3],
                              i_noise=params[4],
                              cs_noise=params[5],
                              scale=1.0)

    def get_residue_params(self, params):
        residue_params = self.reference.copy()
        residue_params['dw'] = params[6:]
        return residue_params

# class RefpeakProblem(ComplexProblem):
#     def get_global_params(self, params):
#         return self.base_globals

#     def get_residue_params(self, params):
#         residue_params = self.reference.copy()

#         residue_params = pd.DataFrame({'residue': self.residues,
#                                         '15N_ref': params[:int(len(params)/3)],
#                                         '1H_ref': params[int(len(params)/3):2*int(len(params)/3)],
#                                         'I_ref': params[2*int(len(params)/3):],
#                                         'dw': self.l1_model[self.gvs:]})
#         residue_params['dw'] = params[6:]
#         return residue_params