import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats


# Returns L1 norm of a vector scaled by the given factor.
def regularization_penalty(lam, vector):
    return lam * np.linalg.norm(vector, 1)

def merged_residue_df(observed_df, reference_df):
    return pd.merge(observed_df, reference_df, on='residue')


class MaliceOptimizer(object):

    def __init__(self, larmor=500, gvs=6, lam=0.0, cs_dist='gaussian',
                 nh_scale=0.2, data=None, mode=None, l1_model=None, 
                 ml_model=None, reference=None):
        self.larmor = larmor
        self.gvs = gvs
        self.lam = lam
        self.nh_scale = nh_scale
        self.data = data
        self.mode = mode
        self.cs_dist = cs_dist
        self.l1_model = l1_model
        self.ml_model = ml_model
        self.reference = reference

        self.residues = list(self.data.groupby('residue').groups.keys())

        if reference is None:
            self.reference = pd.DataFrame()
            for res in self.residues:
                resdata = self.data.copy()[self.data.residue == res]
                self.reference = self.reference.append(resdata.loc[resdata.titrant == np.min(resdata.titrant), ['residue', '15N', '1H', 'intensity']].mean(axis=0), ignore_index=True)
            self.reference = self.reference.rename(columns={'intensity': 'I_ref',
                                                            '15N': '15N_ref',
                                                            '1H': '1H_ref'})

    def delta_ws(self):
        if self.ml_model is None:
            return None
        return self.ml_model[self.gvs:]/self.larmor

    def set_bounds(self, bds):
        self.bounds = bds

    def get_bounds(self):
        return self.bounds

    def get_scipy_bounds(self):
        return tuple([(self.bounds[0][x], self.bounds[1][x]) for x in range(len(self.bounds[0]))])

    # Returns fits for chemical shift and intensity!
    def compute_fits(self, Kd_exp, koff_exp, dR2, amp_scaler, df):
        Kd = np.power(10, Kd_exp)
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
        ihat = df.I_ref/(pa + pb + df.I_ref*(pb*dR2 + i_broad)/amp_scaler)
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

    def add_fits_to_df(self, Kd_exp, koff_exp, dR2, amp_scaler, df):
        cshat, ihat = self.compute_fits(Kd_exp, koff_exp, dR2, amp_scaler, df)
        df['ifit'] = ihat
        df['csfit'] = cshat/self.larmor  # Return as ppm and not Hz

    def add_observed_to_df(self, df):
        csobs = self.observed_chemical_shift(df['15N_ref'], df['15N'], df['1H_ref'], df['1H'])
        df['csp'] = csobs/self.larmor  # Returns as ppm and not Hz

    def pfitter(self):
        Kd_exp, koff_exp, dR2, amp_scaler = self.ml_model[:4]
        residue_params = self.reference.copy()
        residue_params['dw'] = self.ml_model[self.gvs:]
        df = merged_residue_df(self.data, residue_params)

        self.add_fits_to_df(Kd_exp, koff_exp, dR2, amp_scaler, df)
        self.add_observed_to_df(df)
        return df

    # Temporary name. Will be split into multiple functions.
    def enhanced_df(self, params=None):
        if params is None:
            params = list(self.ml_model)

        # Output the per-residue fits
        Kd_exp, koff_exp, dR2, amp_scaler = params[:4]

        concs = self.data[['titrant', 'visible']].drop_duplicates()
        titrant_visible_lm = stats.linregress(concs.titrant, concs.visible)

        if self.mode == 'lfitter':
            titrant_rng = np.max(concs.titrant) - np.min(concs.titrant)
            titrant_vals = np.linspace(np.min(concs.titrant)-0.1*titrant_rng,
                                        np.max(concs.titrant)+0.1*titrant_rng,
                                        1000)
            visible_vals = titrant_visible_lm.slope*titrant_vals + titrant_visible_lm.intercept
            fitter_input = pd.DataFrame({'titrant': titrant_vals,
                                            'visible': visible_vals})
            res_df = pd.DataFrame(itertools.product(titrant_vals, self.residues),
                                    columns=['titrant', 'residue'])
            fitter_input = pd.merge(fitter_input, res_df, on='titrant')

        elif self.mode == 'simulated_peak_generation':
            titrant_vals = concs.titrant.unique()
            visible_vals = titrant_visible_lm.slope*titrant_vals + titrant_visible_lm.intercept
            fitter_input = pd.DataFrame({'titrant': titrant_vals,
                                            'visible': visible_vals})
            res_df = pd.DataFrame(itertools.product(titrant_vals, self.residues),
                                    columns=['titrant', 'residue'])
            pts_15N = []
            pts_1H = []
            pts_int = []
            for i in range(len(res_df)):
                focal_data = self.data[(self.data.residue == res_df.residue[i]) &
                                        (self.data.titrant == res_df.titrant[i])]
                if len(focal_data) >= 1:
                    pts_15N.append(float(focal_data['15N'].mean()))
                    pts_1H.append(float(focal_data['1H'].mean()))
                    pts_int.append(float(focal_data['intensity'].mean()))
                else:
                    pts_15N.append(np.nan)
                    pts_1H.append(np.nan)
                    pts_int.append(np.nan)
            res_df['15N'] = pts_15N
            res_df['1H'] = pts_1H
            res_df['intensity'] = pts_int
            fitter_input = pd.merge(fitter_input, res_df, on='titrant')

        residue_params = self.reference.copy()
        residue_params['dw'] = params[self.gvs:]

        df = merged_residue_df(fitter_input, residue_params)

        self.add_fits_to_df(Kd_exp, koff_exp, dR2, amp_scaler, df)

        if self.mode != "lfitter":
            self.add_observed_to_df(df)
        
        return df

    def fitness(self, params=None):
        if self.mode == 'ml_optimization':
            Kd_exp, koff_exp, dR2, amp_scaler, i_noise, cs_noise = params[:self.gvs]
            residue_params = self.reference.copy()
            residue_params['dw'] = params[self.gvs:]

        elif self.mode == 'reference_optimization':
            Kd_exp, koff_exp, dR2, amp_scaler, i_noise, cs_noise = self.l1_model[:self.gvs]
            residue_params = pd.DataFrame({'residue': self.residues,
                                           '15N_ref': params[:int(len(params)/3)],
                                           '1H_ref': params[int(len(params)/3):2*int(len(params)/3)],
                                           'I_ref': params[2*int(len(params)/3):],
                                           'dw': self.l1_model[self.gvs:]})

        elif self.mode == 'dw_scale_optimization':
            Kd_exp, koff_exp, dR2, amp_scaler, i_noise, cs_noise, scale = params[:self.gvs+1]
            residue_params = self.reference.copy()
            residue_params['dw'] = self.l1_model[self.gvs:]*scale

        else:
            raise NotImplementedError("Unsupported optimization mode " + str(self.mode))


        df = merged_residue_df(self.data, residue_params)

        cshat, ihat = self.compute_fits(Kd_exp, koff_exp, dR2, amp_scaler, df)
        csobs = self.observed_chemical_shift(df['15N_ref'], df['15N'], df['1H_ref'], df['1H'])

        # Compute the likelihoods
        logLL_int = np.sum(stats.norm.logpdf(df.intensity, loc=ihat, scale=i_noise))

        if self.cs_dist == 'gaussian':
            logLL_cs = np.sum(stats.norm.logpdf(csobs, loc=cshat, scale=cs_noise))
        elif self.cs_dist == 'rayleigh':
            logLL_cs = np.sum(stats.rayleigh.logpdf(np.abs(csobs-cshat)/cs_noise) - np.log(cs_noise))
        else:
            print('INVALID CS DISTRIBUTION')
            return 0

        negLL = -1*(logLL_int + logLL_cs - regularization_penalty(self.lam, df.dw))

        return(negLL, )
