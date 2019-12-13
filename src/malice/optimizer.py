import numpy as np
import pandas as pd
import scipy.stats as stats
import itertools

class MaliceOptimizer(object):

    def __init__(self, larmor=500,gvs=7,lam=.01,cs_dist='gaussian',mleinput=None,resgrouped=None,residues=None,mode=None):
        self.larmor = larmor
        self.gvs = gvs
        self.lam = lam
        self.mleinput = mleinput
        self.resgrouped = resgrouped
        self.residues = residues
        self.mode = mode
        self.cs_dist = cs_dist
        self.i = 0
        self.pygmo = False
    
    def set_bounds(self, bds):
        self.bounds = bds
        
    def get_bounds(self):
        return self.bounds
    
    def get_scipy_bounds(self):
        return tuple([(self.bounds[0][x],self.bounds[1][x]) for x in range(len(self.bounds[0]))])
    
    def fitness(self, params=None):
        if self.mode == 'global+dw':
            Kd_exp, koff_exp, dR2, amp, nh_scale, i_noise, cs_noise = params[:self.gvs]
            resparams = self.resgrouped.copy().rename(columns={'intensity':'I_ref','15N':'15N_ref','1H':'1H_ref'})
            resparams['dw'] = params[self.gvs:]
        elif self.mode == 'refpeak_opt':
            Kd_exp, koff_exp, dR2, amp, nh_scale, i_noise, cs_noise = self.model1[:self.gvs]
            resparams = pd.DataFrame({'residue':self.residues,'15N_ref':params[:int(len(params)/3)],
                                      '1H_ref':params[int(len(params)/3):2*int(len(params)/3)],
                                      'I_ref':params[2*int(len(params)/3):],'dw':self.model1[self.gvs:]})
        elif self.mode == 'dw_scale':
            Kd_exp, koff_exp, dR2, amp, nh_scale, i_noise, cs_noise, scale = params[:self.gvs+1]
            resparams = pd.DataFrame({'residue':self.residues,
                                      '15N_ref':self.model2[:int(len(self.model2)/3)], 
                                      '1H_ref':self.model2[int(len(self.model2)/3):2*int(len(self.model2)/3)],
                                      'I_ref':self.model2[2*int(len(self.model2)/3):]})
            resparams['dw'] = self.model1[self.gvs:]*scale
        elif self.mode == 'final_opt':
            Kd_exp, koff_exp, dR2, amp, nh_scale, i_noise, cs_noise = params[:self.gvs]
            resparams = pd.DataFrame({'residue':self.residues,
                                      '15N_ref':self.model2[:int(len(self.model2)/3)], 
                                      '1H_ref':self.model2[int(len(self.model2)/3):2*int(len(self.model2)/3)],
                                      'I_ref':self.model2[2*int(len(self.model2)/3):],
                                      'dw':params[self.gvs:]})
        elif self.mode in ['pfitter','lfitter']:
            ## Output the per-residue fits
            Kd_exp, koff_exp, dR2, amp, nh_scale, i_noise, cs_noise = self.model3[:self.gvs]
            
            concs = self.mleinput[['tit','obs']].drop_duplicates()
            tit_obs_lm = stats.linregress(concs.tit,concs.obs)

            tit_rng = np.max(concs.tit) - np.min(concs.tit)
            tit_vals = np.linspace(np.min(concs.tit)-0.1*tit_rng,
                                   np.max(concs.tit)+0.1*tit_rng,
                                   1000)
            obs_vals = tit_obs_lm.slope*tit_vals + tit_obs_lm.intercept

            fitter_input = pd.DataFrame({'tit':tit_vals,'obs':obs_vals})
            res_df = pd.DataFrame(itertools.product(tit_vals,self.residues),
                                  columns=['tit','residue'])
            fitter_input = pd.merge(fitter_input,res_df,on='tit')

            resparams = pd.DataFrame({'residue':self.residues,
                                      '15N_ref':self.model2[:int(len(self.model2)/3)], 
                                      '1H_ref':self.model2[int(len(self.model2)/3):2*int(len(self.model2)/3)],
                                      'I_ref':self.model2[2*int(len(self.model2)/3):],
                                      'dw':self.model3[self.gvs:]})

        else:
            print('UNSUPPORTED OPTIMIZATION MODE')
            return 0
        
        if self.mode == 'lfitter':   df = pd.merge(fitter_input, resparams,on='residue')
        else:   df = pd.merge(self.mleinput,resparams,on='residue')
        
        Kd = np.power(10,Kd_exp)
        koff = np.power(10,koff_exp)
        kon = koff/Kd
        
        dimer = ( (df.obs + df.tit + Kd) - np.sqrt( np.power((df.obs + df.tit + Kd),2) - 4*df.obs*df.tit ) )/2
        pb = dimer/df.obs
        pa = 1 - pb
        
        #Assume 1:1 stoichiometry for now
        free_tit = df.tit - dimer
        kr = koff
        kf = free_tit*kon
        kex = kr + kf
        
        broad_denom = np.square(np.square(kex) + (1-5*pa*pb)*np.square(df.dw)) + 4*pa*pb*(1-4*pa*pb)*np.power(df.dw,4)
        
        #Compute the fits
        i_broad = pa*pb*np.square(df.dw)*kex * (np.square(kex)+(1-5*pa*pb)*np.square(df.dw))/broad_denom
        ihat = df.I_ref/( pa + pb + df.I_ref*(pb*dR2 + i_broad)/amp )
        cs_broad = pa*pb*(pa-pb)*np.power(df.dw,3) * (np.square(kex)+(1-3*pa*pb)*np.square(df.dw))/broad_denom
        cshat = pb*df.dw - cs_broad
        
        if self.mode == 'lfitter':
            df['ifit'] = ihat
            df['csfit'] = cshat
            return df
            
        csobs = self.larmor*(np.sqrt( np.square(nh_scale*(df['15N'] - df['15N_ref'])) + np.square(df['1H'] - df['1H_ref']) ))
        
        if self.mode == 'pfitter':
            df['csp'] = csobs
            df['ifit'] = ihat
            df['csfit'] = cshat
            return df
    
        
        #Compute the likelihoods
        logLL_int = np.sum( stats.norm.logpdf(df.intensity, loc=ihat, scale=i_noise) )
        
        if self.cs_dist == 'gaussian':
            logLL_cs = np.sum( stats.norm.logpdf(csobs, loc=cshat, scale=cs_noise) )
        elif self.cs_dist == 'rayleigh':
            logLL_cs = np.sum ( stats.rayleigh.logpdf(np.abs(csobs-cshat)/cs_noise) - np.log(cs_noise) )
        else:
            print('INVALID CS DISTRIBUTION')
            return 0
        
        negLL = -1*(logLL_int + logLL_cs - self.lam*np.sum(np.abs(df.dw)))
        
        if self.pygmo:  return(negLL, )
        else:   return(negLL)

    def derivative(self, ):
        ## Return the derivative for each variable in the 
        return 0



    def counter_factory(self):

        if self.mode == 'global+dw':
            def counter(xk, convergence=1e-7):
                if self.i%1000 == 0:
                    print(str(self.i).ljust(8)+'Score: '+str(round(self.fitness(xk),2)).ljust(12)+
                          '-logL: '+str(round(self.fitness(xk)-self.lam*np.sum(xk[self.gvs:]),2)).ljust(12)+
                          'Kd: '+str(round(np.power(10,xk[0]),1)).ljust(10)+
                          'dR2: '+str(round(xk[2],2)).ljust(8)+
                          'max_dw: '+str(round(np.max(xk[self.gvs:]),2)))
                self.i+=1
            return counter
        elif self.mode == 'refpeak_opt':
            def counter(xk,convergence=1e-7):
                if self.i%1000 == 0:
                    print(str(self.i).ljust(8)+'-logL: '+str(round(self.fitness(xk),2)).ljust(12))
                self.i+=1
            return counter
        else:
            print('INVALID MODE FOR COUNTER')
            return 0
