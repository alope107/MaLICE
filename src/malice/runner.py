### MaLICE v190923-0903
### First standalone script version

## Notable features:    - Split the global variable and delta_w optimization
##                          from reference peak optimization
##                        - Currently using additive C
##                            - Need to go back validate this is best

## Import libraries

import os, sys, itertools, time, datetime, concurrent, multiprocessing, copy
import numpy as np, pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize,basinhopping,differential_evolution
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import concurrent.futures
import pygmo as pg

from malice.optimizer import MaliceOptimizer

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', 
                        type=str, 
                        help='path to the CSV to import')
    parser.add_argument('--pop_size', 
                        type=int,
                        help='Population size to perform differential evolution on',
                        default=20)
    parser.add_argument('--least_squares_max_iter', #TODO: Different maxes for different phases?
                        type=int,
                        help='Maximum number of iterations to run sequential least squares minimization',
                        default=100000)
    parser.add_argument('--bootstraps',
                        type=int,
                        help='Number of bootstraps to perform',
                        default=20)
    parser.add_argument('--confidence',
                        type=float,
                        help='Confidence interval to report for parameter estimates',
                        default=0.8)
    parser.add_argument('--larmor',
                        type=int,
                        help="Larmor frequency (MHz) of 1H in the given magnetic field.",
                        default=500)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to store output files. Creates if non-existent.',
                        default="output")
    parser.add_argument('--phase1_islands',
                        type=int,
                        help='PyGMO phase 1 number of island/populations to generate',
                        default=10)
    parser.add_argument('--phase1_generations',
                        type=int,
                        help='PyGMO phase 1 generations per evolution cycle',
                        default=1000)
    parser.add_argument('--phase1_evo_rounds',
                        type=int,
                        help='PyGMO phase 1 rounds of evolution',
                        default=20)
    parser.add_argument('--phase2_islands',
                        type=int,
                        help='PyGMO phase 2 number of island/populations to generate',
                        default=10)
    parser.add_argument('--phase2_generations',
                        type=int,
                        help='PyGMO phase 2 generations per evolution cycle',
                        default=1000)
    parser.add_argument('--phase2_evo_rounds',
                        type=int,
                        help='PyGMO phase 2 rounds of evolution',
                        default=10)
    parser.add_argument('--bootstrap_generations',
                        type=int,
                        help='PyGMO number of generations per bootstrap',
                        default=2000)
    parser.add_argument('--seed',
                        type=int,
                        help='initial seed for PyGMO (not deterministic, but gets closer)',
                        default=1337)
    parser.add_argument('--tolerance',
                        type=float,
                        help='PyGMO tolerance for both ftol and xtol',
                        default='1e-8')
    # TODO: validate arguments
    return parser.parse_args()

def gen_pop1(mleinput,residues,larmor):
    Kd_exp_random = list(np.random.random(1)*5-1)   # Will span from 100 nM to 10 mM
    kex_exp_random = list(np.random.random(1)*4+3)  # Will span from 1 kHz to 10 Mhz
    dR2_random = list(np.random.random(1)*200)            # 0 - 200 Hz
    amp_random = [np.random.normal(np.mean(mleinput.intensity),np.std(mleinput.intensity)) * 20]
        # random amp logic is that since amp = intensity * lw, lets just randomly sample something from the reasonable intensity
        # range and multiply by 20, which is probably a decent enough guess of typical protein linewidths
    i_noise_random = list(np.mean(mleinput.intensity)/(np.random.random(1)*46+4)) # 1/4 to 1/50th of mean intensity
    cs_noise_random = list(larmor/(np.random.random(1)*4450+50)) # larmor / 50-4500 -- rough range of digital res

    dw_random = list( 0.1*larmor * np.random.random(len(residues)) ) ## Every delta_w is 0-0.1 ppm CSP

    return Kd_exp_random + kex_exp_random + dR2_random + amp_random + i_noise_random + cs_noise_random + dw_random

def gen_pop2(optimizer,resgrouped,residues):
    N_random = list( np.array(resgrouped['15N']) + np.random.normal(0,optimizer.model1[5]/optimizer.nh_scale,len(residues)) )
    H_random = list( np.array(resgrouped['1H']) + np.random.normal(0,optimizer.model1[5],len(residues)) )  
    Iref_random = list( np.array(resgrouped['intensity']) + np.random.normal(0,optimizer.model1[4],len(residues)) )

    return N_random + H_random + Iref_random
    
def gen_bspop(optimizer,residues,gvs=6):
    Kd_exp_pert = [optimizer.model3[0] + np.random.normal(0,0.1)]
    kex_exp_pert = [optimizer.model3[1] + np.random.normal(0,0.1)]
    dR2_pert = [optimizer.model3[2] + np.random.normal(0,1)]
    amp_pert = [optimizer.model3[3] + np.random.normal(0,optimizer.model3[3]/20)]
    i_noise_pert = [optimizer.model3[4] + np.random.normal(0,optimizer.model3[4]/20)]
    cs_noise_pert = [optimizer.model3[5] + np.random.normal(0,optimizer.model3[5]/20)]
    
    dw_pert = list(np.array(optimizer.model3[gvs:]) + np.random.normal(0,4,len(residues)))

    return Kd_exp_pert + kex_exp_pert + dR2_pert + amp_pert + i_noise_pert + cs_noise_pert + dw_pert

class Reference_Step(object):
        def __init__(self, stepsize=0.5, optimizer=None,gvs=6,nh_scale=0.2):
            self.stepsize = stepsize
            self.optimizer = optimizer
            self.gvs = gvs
            self.nh_scale = nh_scale
        def __call__(self, x):
            s = self.stepsize
            x[0] += s*np.random.normal(0,self.optimizer.model1[self.gvs+5]/self.nh_scale)
            x[1] += s*np.random.normal(0,self.optimizer.model1[self.gvs+5])
            x[2] += s*np.random.normal(0,self.optimizer.model1[self.gvs+4])
            return x



def null_calculator(fx, config, mleinput, model, df, gvs, bds, res):
    dfx = df.copy()
    dfx.loc[dfx.residue == res,'dw'] = 0
    
    mininit = [model[0]-0.2, model[1]-0.2, model[2]-10, model[3]/1.2,
               model[4]/1.2, model[5]/1.2, model[6]/1.2] + list(dfx.dw/2)
    maxinit = [model[0]+0.2, model[1]+0.2, model[2]+10, model[3]*1.2,
               model[4]*1.2, model[5]*1.2, model[6]*1.2] + list(dfx.dw*2)
    
    for i in range(gvs):
        if mininit[i] < bds[i][0]:    mininit[i] = bds[i][0]
        if maxinit[i] > bds[i][1]:    maxinit[i] = bds[i][1]
    
    bdsx = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])
    
    initx = list(model)
    
    # Minimize
    nullLL = minimize(fx, initx, args=(mleinput,), method='SLSQP', bounds=bdsx,
                      tol=1e-7, options={'disp':True,'maxiter':config.least_squares_max_iter})

    print('null LogL calculated for residue '+str(res))
    
    return nullLL.fun

def main():
    args = _parse_args()
    make_output_dir(args.output_dir)
    np.random.seed(seed= args.seed)
    run_malice(args)
    
def make_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def parse_input(fname, larmor, nh_scale):
    input = pd.read_csv(fname,
                        dtype = {'residue':np.int64,'15N':np.float64,
                                 '1H':np.float64,'intensity':np.float64,
                                 'tit':np.float64,'obs':np.float64})
    mleinput = input.copy()
    
    residues = list(mleinput.groupby('residue').groups.keys())
    resgrouped = pd.DataFrame()
    for res in residues:
        resdata = mleinput.copy()[mleinput.residue == res]
        resgrouped = resgrouped.append(resdata.loc[resdata.intensity == np.max(resdata.intensity),['residue','15N','1H','intensity']])
    
    return mleinput, resgrouped, residues

def run_malice(config):
    starttime = time.time()
    
    fname_prefix = config.input_file.split('/')[-1].split('.')[0]
    
    ## Important variables
    larmor = config.larmor
    gvs = 6
    lam = 0.01
    nh_scale = 0.2  # Consider reducing to ~0.14
    bootstraps = config.bootstraps
    
    tol = config.tolerance
    
    pygmo_seed = 2*config.seed - 280
    pg.set_global_rng_seed(seed = pygmo_seed)
    
    mleinput, resgrouped, residues = parse_input(config.input_file, larmor, nh_scale)
    
    i_noise_est = np.mean(mleinput.intensity)/10
    
    optimizer = MaliceOptimizer(larmor=larmor, 
                                gvs=gvs, 
                                lam=lam,
                                data = mleinput,  
                                #resgrouped=resgrouped,
                                #residues=residues, 
                                mode='global+dw',
                                cs_dist='gaussian',
                                nh_scale = nh_scale)
    
    mininit1 = [-1, 1, 0, np.min(mleinput.intensity)/10, i_noise_est/10, larmor/4500] + list([0]*len(residues))
    maxinit1 = [4, 7, 200, np.max(mleinput.intensity)*200, i_noise_est*10, larmor/50] + list([6*larmor]*len(residues))
    optimizer.set_bounds((mininit1,maxinit1))

    print('\n---  Round 1: initial global variable and delta w optimization  ---\n')
    
    optimizer.pygmo = True
    
    archi = pg.archipelago(prob = pg.problem(optimizer),
                           s_pol = pg.select_best(0.10),
                           r_pol = pg.fair_replace(0.05),
                           t = pg.fully_connected(),
                           seed = pygmo_seed+25)
    archi.set_migration_type(pg.migration_type.broadcast)
    for iteration in range(config.phase1_islands):
        pop = pg.population(pg.problem(optimizer))
        for x in range(config.pop_size):    pop.push_back( gen_pop1(mleinput,residues,larmor) )
        archi.push_back(pop = pop, algo = pg.sade(gen=config.phase1_generations,variant=6,variant_adptv=2,ftol=tol,xtol=tol, seed=pygmo_seed+10*(iteration+1)))
    print(archi)
    archi.evolve(config.phase1_evo_rounds)
    archi.wait()
    best_score = np.array(archi.get_champions_f()).min()
    print(best_score)
    best_index = archi.get_champions_f().index(best_score)
    optimizer.model1 = archi.get_champions_x()[best_index]
    
    print('\n\tKd = '+str(round(np.power(10,optimizer.model1[0]),2))+
          '\n\tkoff = '+str(round(np.power(10,optimizer.model1[1]),2))+
          '\n\tdR2 = '+str(round(optimizer.model1[2],2))+
          '\n\tAmp = '+str(round(optimizer.model1[3],2))+
          '\n\tInoise = '+str(round(optimizer.model1[4],2))+
          '\n\tCSnoise = '+str(round(optimizer.model1[5],2))+
          '\n\tMax dw = '+str(round(np.max(optimizer.model1[gvs:]),2)))
    
    print('Unpenalized -logL = '+str(round(optimizer.fitness(optimizer.model1)[0] - lam*np.sum(optimizer.model1[gvs:]),2)))
    
    runtime = time.time()-starttime
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=runtime)).split('.')[0])
    
    
    ## Stage 2 - reference peak optimization
    print('\n---  Round 2: reference peak optimization  ---\n')
    
    optimizer.mode = 'refpeak_opt'
    optimizer.cs_dist = 'rayleigh'
    
    
    mininit2 =  list(np.array(resgrouped['15N']) - 0.2) + list(
                     np.array(resgrouped['1H']) - 0.05) + list(np.array(resgrouped.intensity)/10) 
    maxinit2 =  list(np.array(resgrouped['15N']) + 0.2) + list(
                     np.array(resgrouped['1H']) + 0.05) + list(np.array(resgrouped.intensity)*10) 
    optimizer.set_bounds((mininit2,maxinit2))
    
    pre_ref = optimizer.reference.copy()
    
    archi = pg.archipelago(prob = pg.problem(optimizer),
                           s_pol = pg.select_best(0.10),
                           r_pol = pg.fair_replace(0.05),
                           t = pg.fully_connected(),
                           seed=pygmo_seed-25)
    archi.set_migration_type(pg.migration_type.broadcast)
    for iteration in range(config.phase2_islands):
        pop = pg.population(pg.problem(optimizer))
        for x in range(config.pop_size):    pop.push_back( gen_pop2(optimizer,resgrouped,residues) )
        archi.push_back(pop = pop, algo = pg.sade(gen=config.phase2_generations,variant=6,variant_adptv=2,ftol=tol,xtol=tol, seed=pygmo_seed-10*(iteration+1)))
    print(archi)
    archi.evolve(config.phase2_evo_rounds)
    archi.wait()
    best_score = np.array(archi.get_champions_f()).min()
    print(best_score)
    best_index = archi.get_champions_f().index(best_score)
    opt_ref = archi.get_champions_x()[best_index]
    optimizer.reference = pd.DataFrame({'residue':residues,
                                        '15N_ref':opt_ref[:int(len(opt_ref)/3)],
                                        '1H_ref':opt_ref[int(len(opt_ref)/3):int(2*len(opt_ref)/3)],
                                        'I_ref':opt_ref[int(2*len(opt_ref)/3):]})
    
    for res in residues:
        nref = float(optimizer.reference.loc[optimizer.reference.residue == res, '15N_ref'])
        href = float(optimizer.reference.loc[optimizer.reference.residue == res, '1H_ref'])
        iref = float(optimizer.reference.loc[optimizer.reference.residue == res, 'I_ref'])
        
        norig = float(pre_ref.loc[pre_ref.residue == res, '15N_ref'])
        horig = float(pre_ref.loc[pre_ref.residue == res, '1H_ref'])
        iorig = float(pre_ref.loc[pre_ref.residue == res, 'I_ref'])
        
        print('Residue '+str(res)+'\t15N: '+str(round(nref,3))+' ('+str(round(nref-norig,3))+
              ')\t1H: '+str(round(href,3))+' ('+str(round(href-horig,3))+
              ')\tIntensity: '+str(round(iref))+' ('+str(round(iref-iorig,3))+')')
    
    
    runtime = time.time()-starttime
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=runtime)).split('.')[0])
    
    
    ## Stage 3 - polish off the model with gradient minimization
    print('\n---  Round 3: gradient minimization of global variables and delta w  ---\n')
    
    optimizer.lam=0
    optimizer.mode = 'dw_scale'
    
    # Bounds

    mininit3a = [optimizer.model1[0]-1, optimizer.model1[1]-1, optimizer.model1[2]-20, optimizer.model1[3]/4, 
                 optimizer.model1[4]/4, optimizer.model1[5]/4, 0.1]
    maxinit3a = [optimizer.model1[0]+1, optimizer.model1[1]+1, optimizer.model1[2]+20, optimizer.model1[3]*4, 
                 optimizer.model1[4]*4, optimizer.model1[5]*4, 10]
    # Fix any of the global bounds that go off into stupid places
    for i in range(gvs):
        if mininit3a[i] < mininit1[i]:    mininit3a[i] = mininit1[i]
        if maxinit3a[i] > maxinit1[i]:    maxinit3a[i] = maxinit1[i]
    optimizer.set_bounds((mininit3a,maxinit3a))

    ## Run the 3a scaling optimization

    init3a = list(optimizer.model1[:gvs]) + [1]

    model3a = minimize(optimizer.fitness, init3a, method='SLSQP',bounds=optimizer.get_scipy_bounds(),
                       tol=1e-7,options={'disp':True,'maxiter':config.least_squares_max_iter,'ftol':1e-7})


    ## Run the 3b fine tuning optimization
    optimizer.mode = 'final_opt'
    # Bounds

    mininit3b = [model3a.x[0]-0.2, model3a.x[1]-0.2, model3a.x[2]-10, model3a.x[3]/1.2,
                 model3a.x[4]/1.2, model3a.x[5]/1.2] + [0]*len(residues)
    maxinit3b = [model3a.x[0]+0.2, model3a.x[1]+0.2, model3a.x[2]+10, model3a.x[3]*1.2,
                 model3a.x[4]*1.2, model3a.x[5]*1.2] + list(np.array(optimizer.model1[gvs:])*model3a.x[-1]*2)
    # Fix any of the global bounds that go off into stupid places
    for i in range(gvs):
        if mininit3b[i] < mininit1[i]:    mininit3b[i] = mininit1[i]
        if maxinit3b[i] > maxinit1[i]:    maxinit3b[i] = maxinit1[i]
    #optimizer.set_bounds((mininit3b,maxinit3b))
    optimizer.set_bounds((mininit1,maxinit1))

    init3b = list(model3a.x[:gvs]) + list(np.array(optimizer.model1[gvs:])*model3a.x[-1])
    
    # Full minimization
    model3b = minimize(optimizer.fitness, init3b, method='SLSQP', bounds=optimizer.get_scipy_bounds(),
                      tol=1e-7, options={'disp':True,'maxiter':config.least_squares_max_iter,'ftol':1e-7})

    print('\nFinal Score = '+str(round(model3b.fun,2)))
    
    optimizer.model3 = model3b.x

    ## Let's go ahead and print some results + save a figure
    print('\n\tKd = '+str(round(np.power(10,optimizer.model3[0]),2))+
          '\n\tkoff = '+str(round(np.power(10,optimizer.model3[1]),2))+
          '\n\tdR2 = '+str(round(optimizer.model3[2],2))+
          '\n\tAmp = '+str(round(optimizer.model3[3],2))+
          '\n\tInoise = '+str(round(optimizer.model3[4],2))+
          '\n\tCSnoise = '+str(round(optimizer.model3[5],2))+
          '\n\tMax dw = '+str(round(np.max(optimizer.model3[gvs:]),2)))

    dfs = pd.DataFrame({'residue':residues,'dw':optimizer.model3[gvs:]/larmor})
    ## Print out data
    csv_name = os.path.join(config.output_dir, fname_prefix + '_MaLICE_fits.csv')
    txt_name = os.path.join(config.output_dir, fname_prefix+'_MaLICE_deltaw.txt')
    dfs.to_csv(csv_name,index=False)
    dfs[['residue','dw']].to_csv(txt_name, index=False,header=False)
    

    ## Output the per-residue fits
    
    optimizer.mode = 'lfitter'
    fit_data = optimizer.fitness()
    optimizer.mode = 'pfitter'
    mleoutput = optimizer.fitness()
    
    tit_rng = fit_data.tit.max()-fit_data.tit.min()
    xl = [np.min(fit_data.tit)-0.01*tit_rng, np.max(fit_data.tit)+0.01*tit_rng]
    csp_rng = np.max(mleoutput.csp)-np.min(mleoutput.csp)
    yl_csp = [np.min(mleoutput.csp)-0.05*csp_rng, np.max(mleoutput.csp)+0.05*csp_rng]
    int_rng = np.max(mleoutput.intensity)-np.min(mleoutput.intensity)
    yl_int = [np.min(mleoutput.intensity)-0.05*int_rng, np.max(mleoutput.intensity)+0.05*int_rng]

    pdf_name = os.path.join(config.output_dir, fname_prefix + '_MaLICE_fits.pdf')
    with PdfPages(pdf_name) as pdf:
        for residue in residues:
            fig, ax = plt.subplots(ncols=2,figsize=(7.5,2.5))
            ax[0].scatter('tit','csp',data=mleoutput[mleoutput.residue == residue],color='black',s=10)
            ax[0].errorbar('tit','csp',data=mleoutput[mleoutput.residue == residue],yerr=optimizer.model3[5]/larmor,color='black',fmt='none',s=16)
            ax[0].plot('tit','csfit',data=fit_data[fit_data.residue == residue])
            ax[1].scatter('tit','intensity',data=mleoutput[mleoutput.residue == residue],color='black',s=10)
            ax[1].errorbar('tit','intensity',data=mleoutput[mleoutput.residue == residue],yerr=optimizer.model3[4],color='black',fmt='none',s=16)
            ax[1].plot('tit','ifit',data=fit_data[fit_data.residue == residue])
            ax[0].set(xlim=xl, ylim=yl_csp, xlabel='Titrant (μM)', ylabel='CSP (ppm)', title='Residue '+str(residue)+' CSP')
            ax[1].set(xlim=xl, ylim=yl_int, xlabel='Titrant (μM)', ylabel='Intensity', title='Residue '+str(residue)+' Intensity')
            fig.tight_layout()
            pdf.savefig()
            plt.close()
    
    
    
    ## 191127 CODE FOR BOOTSTRAPPING
    if bootstraps > 0:
        min_bs = [optimizer.model3[0]-1, optimizer.model3[1]-1, optimizer.model3[2]-20, optimizer.model3[3]/4, 
                  optimizer.model3[4]/4, optimizer.model3[5]/4] + [0]*len(residues)
        max_bs = [optimizer.model3[0]+1, optimizer.model3[1]+1, optimizer.model3[2]+20, optimizer.model3[3]*4, 
                  optimizer.model3[4]*4, optimizer.model3[5]*4] + list(np.array(optimizer.model3[gvs:])*3)

        archi = pg.archipelago(t = pg.unconnected(),
                               r_pol= pg.fair_replace(0),
                               seed = pygmo_seed+37)
        bs_problems = []
        for x in range(bootstraps):
            bs_opt = MaliceOptimizer(larmor=larmor, 
                                     gvs=gvs,
                                     data = mleinput,
                                     mode='final_opt',
                                     cs_dist='rayleigh',
                                     nh_scale = nh_scale,
                                     model1 = optimizer.model1,
                                     model2 = optimizer.model2,
                                     model3 = optimizer.model3,
                                     bootstrap = True)
            bs_opt.set_bounds((min_bs,max_bs))
            bs_problems.append(bs_opt)
        for x in range(bootstraps):
            pop = pg.population(pg.problem(bs_problems[x]))
            for k in range(config.pop_size):    pop.push_back( gen_bspop(bs_problems[x],residues,gvs) )
            archi.push_back(pop = pop, algo = pg.algorithm( pg.sade(gen=config.bootstrap_generations,variant=6,variant_adptv=2,ftol=tol,xtol=tol,seed=pygmo_seed*(23+x)+49) ) )
        print(archi)
        archi.evolve(1)
        archi.wait()
        bsmodels = archi.get_champions_x()
        
        #Test out reporting of confidence intervals using global parameters
        global_params = ['Kd_exp','koff_exp','dR2','Amp','I_noise','CS_noise']
        
        lower_limit = (1-config.confidence)/2
        lower_bs = int( np.round(lower_limit*bootstraps) )
        upper_limit = 1-lower_limit
        upper_bs = int( np.round(upper_limit*bootstraps) ) - 1
        
        print('EMPIRICAL CONFIDENCE INTERVAL ('+str(round(config.confidence*100))+'%):')
        for k in range(gvs):
            bs_params = [bsmodel[k] for bsmodel in bsmodels]
            bs_delta = np.sort( np.array(bs_params) - optimizer.model3[k] )
            print(bs_delta)
            lower_cl = optimizer.model3[k] - bs_delta[upper_bs]
            if lower_cl < mininit1[k]:  lower_cl = mininit1[k]
            upper_cl = optimizer.model3[k] - bs_delta[lower_bs]
            if upper_cl > maxinit1[k]:  upper_cl = maxinit1[k]
            
            print('\t'+global_params[k]+' = '+str(round(optimizer.model3[k],2))+' ['+str(round(lower_cl,2))+','+str(round(upper_cl,2))+']\n')
        
        print('NORMAL CONFIDENCE INTERVAL (95%):')
        for k in range(gvs):
            bs_params = [bsmodel[k] for bsmodel in bsmodels]
            bs_stderr = np.std(bs_params,ddof=1)
            print('\t'+global_params[k]+' = '+str(round(optimizer.model3[k],2))+' -/+ '+str(round(1.96*bs_stderr,2)))

    ## Perform likelihood ratio tests
    ##### DISABLE ALL OF THESE FOR THE TIME BEING -- NEEDS TO BE REWRITTEN
    '''
    print('\n---  Round 4: likelihood ratio test of parameters  ---\n')
    executor = concurrent.futures.ProcessPoolExecutor(config.thread_count)
    futures = [executor.submit(null_calculator, optimizer.fitness, config, mleinput, model3b.x, dfs, gvs, bds1, r) for r in residues]
    concurrent.futures.wait(futures)

    dfs['altLL'] = -1 * model3b.fun
    dfs['nullLL'] = [-1*x.result() for x in futures]
    dfs['deltaLL'] = dfs.altLL - dfs.nullLL
    dfs['rank'] = dfs['deltaLL'].rank(ascending=False,na_option='top')
    dfs['p-value'] = (1-stats.chi2.cdf(2*dfs.deltaLL,df=1))*len(dfs)/dfs['rank']
    dfs = dfs.sort_values('deltaLL',ascending=False)
    dfs['sig'] = dfs['p-value'] < 0.01
    
    png_name = os.path.join(config.output_dir, fname_prefix + '_MaLICE_plot.png')
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter('residue','dw',data=dfs[dfs.sig == False],color='black',s=80)
    #ax.errorbar('residue','dw',yerr='stderr',data=dfs[dfs.sig == False],color='black',fmt='none',s=20)
    ax.scatter('residue','dw',data=dfs[dfs.sig == True],color='red',s=80)
    #ax.errorbar('residue','dw',yerr='stderr',data=dfs[dfs.sig == True],color='red',fmt='none',s=20)
    ax.set(xlim=(np.min(dfs.residue)+1,np.max(dfs.residue)+1),xlabel='Residue',ylabel='Δω (Hz)')
    fig.savefig(png_name,dpi=600,bbox_inches='tight',pad_inches=0)
    '''


    endtime = time.time()
    runtime = endtime-starttime
    print('\n\nRun time = '+str(datetime.timedelta(seconds=runtime)).split('.')[0])
if __name__ == "__main__":
    main()
