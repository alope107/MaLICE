### MaLICE v200101

## Import libraries

import os, sys, itertools, time, datetime, concurrent, multiprocessing, copy
import numpy as np, pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize,basinhopping,differential_evolution,curve_fit
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import concurrent.futures
import pygmo as pg
from fpdf import FPDF
import nmrglue as ng
import pickle

from malice.optimizer import MaliceOptimizer
from malice.reporter import CompLEx_Report

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
                        default=0)
    parser.add_argument('--confidence',
                        type=float,
                        help='Confidence interval to report for parameter estimates',
                        default=0.95)
    parser.add_argument('--larmor',
                        type=float,
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
                        default=1500)
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
                        default=3000)
    parser.add_argument('--mcmc_walks',
                        type=int,
                        help='Number of MCMC walks to perform for confidence interval calculation',
                        default=50)
    parser.add_argument('--mcmc_steps',
                        type=int,
                        help='Maximum number of steps per MCMC walk for confidence interval calculation',
                        default=10000)
    parser.add_argument('--num_threads',
                        type=int,
                        help='Number of threads to parallelize MCMC walks',
                        default=10)
    parser.add_argument('--seed',
                        type=int,
                        help='initial seed for PyGMO (not deterministic, but gets closer)',
                        default=1337)
    parser.add_argument('--tolerance',
                        type=float,
                        help='PyGMO tolerance for both ftol and xtol',
                        default='1e-8')
    parser.add_argument('--visible',
                        type=str,
                        help='Name of the NMR visible protein',
                        default='Sample protein 15N')
    parser.add_argument('--titrant',
                        type=str,
                        help='Name of the titrant',
                        default='Sample titrant')
    # TODO: validate arguments
    return parser.parse_args()

def gen_pop1(optimizer):
    Kd_exp_random = list(np.random.random(1)*5-1)   # Will span from 100 nM to 10 mM
    kex_exp_random = list(np.random.random(1)*4+3)  # Will span from 1 kHz to 10 Mhz
    dR2_random = list(np.random.random(1)*200)            # 0 - 200 Hz
    amp_scaler_random = [np.random.normal(np.mean(optimizer.data.intensity),np.std(optimizer.data.intensity)) * 20]
        # random amp logic is that since amp = intensity * lw, lets just randomly sample something from the reasonable intensity
        # range and multiply by 20, which is probably a decent enough guess of typical protein linewidths
    i_noise_random = list(np.mean(optimizer.data.intensity)/(np.random.random(1)*46+4)) # 1/4 to 1/50th of mean intensity
    cs_noise_random = list(optimizer.larmor/(np.random.random(1)*4450+50)) # larmor / 50-4500 -- rough range of digital res

    dw_random = list( 0.1*optimizer.larmor * np.random.random(len(optimizer.residues)) ) ## Every delta_w is 0-0.1 ppm CSP

    return Kd_exp_random + kex_exp_random + dR2_random + amp_scaler_random + i_noise_random + cs_noise_random + dw_random

def gen_pop2(optimizer):
    N_random = list( np.array(optimizer.reference['15N_ref']) + np.random.normal(0,optimizer.l1_model[5]/optimizer.nh_scale,len(optimizer.residues)) )
    H_random = list( np.array(optimizer.reference['1H_ref']) + np.random.normal(0,optimizer.l1_model[5],len(optimizer.residues)) )  
    Iref_random = list( np.array(optimizer.reference['I_ref']) + np.random.normal(0,optimizer.l1_model[4],len(optimizer.residues)) )

    return N_random + H_random + Iref_random

def pygmo_wrapper(optimizer,pop_generator,seed,islands,pop_size,generations,evo_rounds,tolerance):
    archi = pg.archipelago(prob = pg.problem(optimizer),
                           s_pol = pg.select_best(0.10),
                           r_pol = pg.fair_replace(0.05),
                           t = pg.fully_connected(),
                           seed = seed)
    archi.set_migration_type(pg.migration_type.broadcast)
    for iteration in range(islands):
        pop = pg.population(pg.problem(optimizer))
        for x in range(pop_size):    pop.push_back( pop_generator(optimizer) )
        archi.push_back(pop = pop, algo = pg.sade(gen=generations,variant=6,variant_adptv=2,ftol=tolerance,xtol=tolerance, seed=seed+10*(iteration+1)))
    archi.evolve(evo_rounds)
    archi.wait()
    best_score = np.array(archi.get_champions_f()).min()
    best_index = archi.get_champions_f().index(best_score)
    best_model = archi.get_champions_x()[best_index]
    
    return best_model, best_score

def gen_bspop(optimizer):
    Kd_exp_pert = [optimizer.ml_model[0] + np.random.normal(0,0.2)]
    kex_exp_pert = [optimizer.ml_model[1] + np.random.normal(0,0.2)]
    dR2_pert = [optimizer.ml_model[2] + np.random.normal(0,1)]
    amp_scaler_pert = [optimizer.ml_model[3] + np.random.normal(0,optimizer.ml_model[3]/10)]
    i_noise_pert = [optimizer.ml_model[4] + np.random.normal(0,optimizer.ml_model[4]/10)]
    cs_noise_pert = [optimizer.ml_model[5] + np.random.normal(0,optimizer.ml_model[5]/10)]
    
    dw_pert = list(np.array(optimizer.ml_model[optimizer.gvs:]) + np.random.normal(0,3,len(optimizer.residues)))

    return Kd_exp_pert + kex_exp_pert + dR2_pert + amp_scaler_pert + i_noise_pert + cs_noise_pert + dw_pert

def bootstrapper(optimizer,seed,pop_size,bootstrap_num,bootstrap_gen,tolerance):
    min_bs = [optimizer.ml_model[0]-1, optimizer.ml_model[1]-1, optimizer.ml_model[2]-20, optimizer.ml_model[3]/4, 
              optimizer.ml_model[4]/4, optimizer.ml_model[5]/4] + [0]*(len(optimizer.ml_model)-optimizer.gvs)
    max_bs = [optimizer.ml_model[0]+1, optimizer.ml_model[1]+1, optimizer.ml_model[2]+20, optimizer.ml_model[3]*4, 
              optimizer.ml_model[4]*4, optimizer.ml_model[5]*4] + list(np.array(optimizer.ml_model[optimizer.gvs:])*3)

    archi = pg.archipelago(t = pg.unconnected(),
                           r_pol= pg.fair_replace(0),
                           seed = 79*seed+3723)
    bs_problems = []
    for x in range(bootstrap_num):
        bs_opt = MaliceOptimizer(larmor=optimizer.larmor, 
                                 gvs=optimizer.gvs,
                                 data = optimizer.data,
                                 mode='ml_optimization',
                                 cs_dist='rayleigh',
                                 nh_scale = optimizer.nh_scale,
                                 l1_model = optimizer.l1_model,
                                 reference = optimizer.reference,
                                 ml_model = optimizer.ml_model,
                                 bootstrap = True)
        bs_opt.set_bounds((min_bs,max_bs))
        bs_problems.append(bs_opt)
    for x in range(bootstrap_num):
        pop = pg.population(pg.problem(bs_problems[x]))
        for k in range(pop_size):    pop.push_back( gen_bspop(bs_problems[x]) )
        archi.push_back(pop = pop, algo = pg.algorithm( pg.sade(gen=bootstrap_gen,variant=6,variant_adptv=2,ftol=tolerance,xtol=tolerance,seed=seed*(23+x)+49) ) )
    archi.evolve(1)
    archi.wait()
    return archi.get_champions_x()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def mcmc_walker(optimizer, steps, confidence, min_global, max_global, seed, iterator=1, abort_threshold = 100):
    
    np.random.seed(seed = 9*seed + 89*iterator)
    
    walker = MaliceOptimizer(larmor = optimizer.larmor,
                             gvs = optimizer.gvs,
                             data = optimizer.data,
                             mode = 'ml_optimization',
                             nh_scale = optimizer.nh_scale,
                             l1_model = optimizer.l1_model,
                             reference = optimizer.reference,
                             ml_model = optimizer.ml_model)
    
    min_mcmc = [walker.ml_model[0]-1, walker.ml_model[1]-1, walker.ml_model[2]-20, walker.ml_model[3]/4, 
                walker.ml_model[4]/4, walker.ml_model[5]/4] + [0]*(len(walker.ml_model)-walker.gvs)
    max_mcmc = [walker.ml_model[0]+1, walker.ml_model[1]+1, walker.ml_model[2]+20, walker.ml_model[3]*4, 
                walker.ml_model[4]*4, walker.ml_model[5]*4] + [max(walker.ml_model[walker.gvs:])*5]*(len(walker.ml_model)-walker.gvs)
    
    # Fix any of the global bounds that go off into stupid places
    for i in range(walker.gvs):
        if min_mcmc[i] < min_global[i]:    min_mcmc[i] = min_global[i]
        if max_mcmc[i] > max_global[i]:    max_mcmc[i] = max_global[i]
    
    tolerated_negLL = walker.fitness(walker.ml_model)[0] + stats.chi2.ppf(confidence,df=1)/2
    
    model = list(walker.ml_model)
    accepted_steps = []
    consecutive_failed = 0
    for i in range(steps):
        prev_model = list(model)
        
        # For global variables, allow it to walk randomly -/+ 0.4% and dw to move randomly -/+ 0.1 Hz
        perturbation = list((np.random.random(walker.gvs)-0.5)/125*np.array(model[:walker.gvs])) + list((np.random.random(len(model[walker.gvs:]))-0.5)/5)
        model = list(np.array(model) + perturbation)
        
        #Check bounds
        for j in range(len(model)):
            if model[j] < min_mcmc[j]:  model[j] = min_mcmc[j]
            if model[j] > max_mcmc[j]:  model[j] = max_mcmc[j]
        
        if walker.fitness(model)[0] <= tolerated_negLL:
            accepted_steps.append(list(model))
            consecutive_failed = 0
        else:
            model = list(prev_model)
            consecutive_failed += 1
        
        if consecutive_failed > abort_threshold:    break
        
    print('Walk #'+str(iterator+1)+': Accepted '+str(len(accepted_steps))+' of '+str(i+1)+' steps')
    
    return accepted_steps



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
                                 'titrant':np.float64,'visible':np.float64})
    data = input.copy()
    
    residues = list(data.groupby('residue').groups.keys())
    reference_points = pd.DataFrame()
    for res in residues:
        resdata = data.copy()[data.residue == res]
        ## Use the lowest titration point (hopefully zero) for the reference
        min_titrant_conc = resdata.titrant.min()
        reference_points = reference_points.append(resdata.loc[resdata.titrant == min_titrant_conc,['residue','15N','1H','intensity']])
        #resgrouped = resgrouped.append(resdata.loc[resdata.intensity == np.max(resdata.intensity),['residue','15N','1H','intensity']])
    
    return data, reference_points, residues

def run_malice(config):
    performance = {}
    performance['start_time'] = time.time()
    
    fname_prefix = config.input_file.split('/')[-1].split('.')[0]
    
    ## Important variables
    larmor = config.larmor
    gvs = 6
    lam = 0.012
    nh_scale = 0.2  # Consider reducing to ~0.14
    
    pygmo_seed = 2*config.seed - 280
    pg.set_global_rng_seed(seed = pygmo_seed)
    
    user_data, initial_reference, residues = parse_input(config.input_file, larmor, nh_scale)
    
    i_noise_est = np.mean(user_data.intensity)/10
    
    optimizer = MaliceOptimizer(larmor=larmor, 
                                gvs=gvs, 
                                lam=lam,
                                data = user_data,
                                mode='ml_optimization',
                                cs_dist='gaussian',
                                nh_scale = nh_scale)
    
    
    
    ## Stage 1 - parameter optimization with L1 regularization
    print('\n---  Phase 1: Differential evolution for parameter optimization with L1 regularization  ---\n')
    
    l1_bounds_min = [-1, 1, 0, np.min(user_data.intensity)/10, i_noise_est/10, larmor/4500] + list([0]*len(residues))
    l1_bounds_max = [4, 7, 200, np.max(user_data.intensity)*200, i_noise_est*10, larmor/50] + list([6*larmor]*len(residues))
    optimizer.set_bounds((l1_bounds_min,l1_bounds_max))
    
    optimizer.l1_model, performance['l1_model_score'] = pygmo_wrapper(optimizer,
                                                                 pop_generator = gen_pop1,
                                                                 seed = pygmo_seed+987,
                                                                 islands = config.phase1_islands,
                                                                 pop_size = config.pop_size,
                                                                 generations = config.phase1_generations,
                                                                 evo_rounds = config.phase1_evo_rounds,
                                                                 tolerance=config.tolerance)
    print('\tL1-penalized -logL:\t'+str(round(performance['l1_model_score'],2)))
    optimizer.lam = 0
    performance['l1_unpenalized_score'] = optimizer.fitness(optimizer.l1_model)[0]
    print('\tUnpenalized -logL:\t'+str(round(performance['l1_unpenalized_score'],2)))
    
    print('\n\tKd = '+str(round(np.power(10,optimizer.l1_model[0]),2))+
          '\n\tkoff = '+str(round(np.power(10,optimizer.l1_model[1]),2))+
          '\n\tdR2 = '+str(round(optimizer.l1_model[2],2))+
          '\n\tAmp_scaler = '+str(round(optimizer.l1_model[3],2))+
          '\n\tInoise = '+str(round(optimizer.l1_model[4],2))+
          '\n\tCSnoise = '+str(round(optimizer.l1_model[5],2))+
          '\n\tMax dw = '+str(round(np.max(optimizer.l1_model[gvs:]),2)))
    
    performance['phase1_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])
    
    
    
    
    ## Stage 2 - reference peak optimization
    print('\n---  Phase 2: Reference peak optimization  ---\n')
    
    optimizer.mode = 'reference_optimization'
    optimizer.cs_dist = 'rayleigh'
    
    ref_opt_bounds_min =  list(np.array(initial_reference['15N']) - 0.2) + list(
                               np.array(initial_reference['1H']) - 0.05) + list(np.array(initial_reference.intensity)/10) 
    ref_opt_bounds_max =  list(np.array(initial_reference['15N']) + 0.2) + list(
                               np.array(initial_reference['1H']) + 0.05) + list(np.array(initial_reference.intensity)*10) 
    optimizer.set_bounds((ref_opt_bounds_min,ref_opt_bounds_max))
    
    pre_ref = optimizer.reference.copy()
    
    opt_ref, performance['opt_ref_score'] = pygmo_wrapper(optimizer,
                                           pop_generator = gen_pop2,
                                           seed = pygmo_seed-1987,
                                           islands = config.phase2_islands,
                                           pop_size = config.pop_size,
                                           generations = config.phase2_generations,
                                           evo_rounds = config.phase2_evo_rounds,
                                           tolerance=config.tolerance)
    optimizer.reference = pd.DataFrame({'residue':residues,
                                        '15N_ref':opt_ref[:int(len(opt_ref)/3)],
                                        '1H_ref':opt_ref[int(len(opt_ref)/3):int(2*len(opt_ref)/3)],
                                        'I_ref':opt_ref[int(2*len(opt_ref)/3):]})
    
    print('\t-logL after reference optimization:\t'+str(round(performance['opt_ref_score'],2)))
    
    # Output the optimized references as a csv
    reference_delta_df = pre_ref.copy().rename(columns={'15N_ref':'15N_original',
                                                        '1H_ref':'1H_original',
                                                        'I_ref':'Intensity_original'})
    reference_delta_df['15N_optimized'] = opt_ref[:int(len(opt_ref)/3)]
    reference_delta_df['1H_optimized'] = opt_ref[int(len(opt_ref)/3):int(2*len(opt_ref)/3)]
    reference_delta_df['Intensity_optimized'] = opt_ref[int(2*len(opt_ref)/3):]
    reference_delta_df['15N_delta'] = reference_delta_df['15N_original'] - reference_delta_df['15N_optimized']
    reference_delta_df['1H_delta'] = reference_delta_df['1H_original'] - reference_delta_df['1H_optimized']
    reference_delta_df['Intensity_delta'] = reference_delta_df['Intensity_original'] - reference_delta_df['Intensity_optimized']
    reference_delta_df = reference_delta_df[['residue','15N_original','15N_optimized','15N_delta',
                                             '1H_original','1H_optimized','1H_delta',
                                             'Intensity_original','Intensity_optimized','Intensity_delta']]
    reference_delta_df = reference_delta_df.round(3)
    reference_csv = os.path.join(config.output_dir, fname_prefix + '_reference_peaks.csv')
    reference_delta_df.to_csv(reference_csv,index=False)
    
    print('\tSaved reference optimization results to '+reference_csv)
    
    performance['phase2_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])
    
    
    
    
    ## Stage 3 - polish off the model with gradient minimization
    print('\n---  Phase 3: Final -logL minimization without regularization  ---\n')
    
    optimizer.mode = 'dw_scale_optimization'
    
    # Bounds
    
    scale_opt_bounds_min = [optimizer.l1_model[0]-1, optimizer.l1_model[1]-1, 
                            optimizer.l1_model[2]-20, optimizer.l1_model[3]/4, 
                            optimizer.l1_model[4]/4, optimizer.l1_model[5]/4, 0.1]
    scale_opt_bounds_max = [optimizer.l1_model[0]+1, optimizer.l1_model[1]+1, 
                            optimizer.l1_model[2]+20, optimizer.l1_model[3]*4, 
                            optimizer.l1_model[4]*4, optimizer.l1_model[5]*4, 10]
    
    # Fix any of the global bounds that go off into stupid places
    for i in range(gvs):
        if scale_opt_bounds_min[i] < l1_bounds_min[i]:    scale_opt_bounds_min[i] = l1_bounds_min[i]
        if scale_opt_bounds_max[i] > l1_bounds_max[i]:    scale_opt_bounds_max[i] = l1_bounds_max[i]
    optimizer.set_bounds((scale_opt_bounds_min,scale_opt_bounds_max))
    
    ## Run the scaling optimization
    #scale_opt_bounds_min = l1_bounds_min[:gvs] + [0.1]
    #scale_opt_bounds_max = l1_bounds_min[:gvs] + [10]
    #optimizer.set_bounds((scale_opt_bounds_min,scale_opt_bounds_max))
    
    scaled_opt_initial = list(optimizer.l1_model[:gvs]) + [1]

    scaled_dw_opt = minimize(optimizer.fitness, scaled_opt_initial, method='SLSQP',bounds=optimizer.get_scipy_bounds(),
                             tol=1e-7,options={'disp':False,'maxiter':config.least_squares_max_iter,'ftol':1e-7})
    
    performance['scaled_dw_score'] = scaled_dw_opt.fun
    dw_scaler = scaled_dw_opt.x[-1]
    print('\n\tOptimized delta_w scaler: '+str(round(dw_scaler,2))+
          '\n\tPost-scalar optimization -logL: '+str(round(performance['scaled_dw_score'],2)))


    ## Run the fine tuning optimization
    optimizer.mode = 'ml_optimization'
    
    # Bounds
    
    ml_opt_bounds_min = [scaled_dw_opt.x[0]-0.2, scaled_dw_opt.x[1]-0.2, scaled_dw_opt.x[2]-10, scaled_dw_opt.x[3]/1.2,
                 scaled_dw_opt.x[4]/1.2, scaled_dw_opt.x[5]/1.2] + [0]*len(residues)
    ml_opt_bounds_max = [scaled_dw_opt.x[0]+0.2, scaled_dw_opt.x[1]+0.2, scaled_dw_opt.x[2]+10, scaled_dw_opt.x[3]*1.2,
                 scaled_dw_opt.x[4]*1.2, scaled_dw_opt.x[5]*1.2] + list(np.array(optimizer.l1_model[gvs:])*scaled_dw_opt.x[-1]*2)
    # Fix any of the global bounds that go off into stupid places
    for i in range(gvs):
        if ml_opt_bounds_min[i] < l1_bounds_min[i]:    ml_opt_bounds_min[i] = l1_bounds_min[i]
        if ml_opt_bounds_max[i] > l1_bounds_max[i]:    ml_opt_bounds_max[i] = l1_bounds_max[i]
    optimizer.set_bounds((ml_opt_bounds_min,ml_opt_bounds_max))
    
    #optimizer.set_bounds((l1_bounds_min,l1_bounds_max))
    
    ml_opt_initial = list(scaled_dw_opt.x[:gvs]) + list(np.array(optimizer.l1_model[gvs:])*dw_scaler)
    
    # Full minimization
    ml_model_opt = minimize(optimizer.fitness, ml_opt_initial, method='SLSQP', 
                            bounds=optimizer.get_scipy_bounds(), tol=1e-7,
                            options={'disp':False,'maxiter':config.least_squares_max_iter,'ftol':1e-7})
    
    performance['ml_model_score'] = ml_model_opt.fun
    print('\n\tFinal -logL: '+str(round(performance['ml_model_score'],2)))
    
    optimizer.ml_model = ml_model_opt.x

    ## Let's go ahead and print some results + save a figure
    print('\n\tKd = '+str(round(np.power(10,optimizer.ml_model[0]),2))+
          '\n\tkoff = '+str(round(np.power(10,optimizer.ml_model[1]),2))+
          '\n\tdR2 = '+str(round(optimizer.ml_model[2],2))+
          '\n\tAmp_scaler = '+str(round(optimizer.ml_model[3],2))+
          '\n\tInoise = '+str(round(optimizer.ml_model[4],2))+
          '\n\tCSnoise = '+str(round(optimizer.ml_model[5],2))+
          '\n\tMax dw = '+str(round(np.max(optimizer.ml_model[gvs:]),2)))

    dfs = pd.DataFrame({'residue':residues,'dw':optimizer.ml_model[gvs:]/larmor})
    ## Print out data
    csv_name = os.path.join(config.output_dir, fname_prefix + '_CompLEx_fits.csv')
    txt_name = os.path.join(config.output_dir, fname_prefix+'_CompLEx_deltaw.txt')
    dfs.to_csv(csv_name,index=False)
    dfs[['residue','dw']].to_csv(txt_name, index=False,header=False)
    
    performance['phase3_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])

    ## Output the per-residue fits
    
    optimizer.mode = 'lfitter'
    fit_data = optimizer.fitness()
    optimizer.mode = 'pfitter'
    mleoutput = optimizer.fitness()
    
    titrant_rng = fit_data.titrant.max()-fit_data.titrant.min()
    xl = [np.min(fit_data.titrant)-0.01*titrant_rng, np.max(fit_data.titrant)+0.01*titrant_rng]
    csp_rng = np.max(mleoutput.csp)-np.min(mleoutput.csp)
    yl_csp = [np.min(mleoutput.csp)-0.05*csp_rng, np.max(mleoutput.csp)+0.05*csp_rng]
    int_rng = np.max(mleoutput.intensity)-np.min(mleoutput.intensity)
    yl_int = [np.min(mleoutput.intensity)-0.05*int_rng, np.max(mleoutput.intensity)+0.05*int_rng]

    pdf_name = os.path.join(config.output_dir, fname_prefix + '_CompLEx_fits.pdf')
    with PdfPages(pdf_name) as pdf:
        for residue in residues:
            fig, ax = plt.subplots(ncols=2,figsize=(7.5,2.5))
            ax[0].scatter('titrant','csp',data=mleoutput[mleoutput.residue == residue],color='black',s=10)
            ax[0].errorbar('titrant','csp',data=mleoutput[mleoutput.residue == residue],yerr=optimizer.ml_model[5]/larmor,color='black',fmt='none',s=16)
            ax[0].plot('titrant','csfit',data=fit_data[fit_data.residue == residue])
            ax[1].scatter('titrant','intensity',data=mleoutput[mleoutput.residue == residue],color='black',s=10)
            ax[1].errorbar('titrant','intensity',data=mleoutput[mleoutput.residue == residue],yerr=optimizer.ml_model[4],color='black',fmt='none',s=16)
            ax[1].plot('titrant','ifit',data=fit_data[fit_data.residue == residue])
            ax[0].set(xlim=xl, ylim=yl_csp, xlabel='Titrant (μM)', ylabel='CSP (ppm)', title='Residue '+str(residue)+' CSP')
            ax[1].set(xlim=xl, ylim=yl_int, xlabel='Titrant (μM)', ylabel='Intensity', title='Residue '+str(residue)+' Intensity')
            fig.tight_layout()
            pdf.savefig()
            plt.close()
    
    
    
    
    
    
    
    
    ## 191127 CODE FOR BOOTSTRAPPING
    if config.bootstraps > 0:
        print('\n---  Phase 4: Confidence interval estimation of global parameters and delta w  ---\n')
        print('\tNumber of bootstrap replicates: '+str(config.bootstraps))
        
        bsmodels = bootstrapper(optimizer = optimizer, seed = pygmo_seed+9228,
                                pop_size = config.pop_size, bootstrap_num = config.bootstraps, 
                                bootstrap_gen = config.bootstrap_generations,
                                tolerance = config.tolerance)
    
        #Test out reporting of confidence intervals using global parameters
        global_params = ['Kd_exp','koff_exp','dR2','Amp_scaler','I_noise','CS_noise']
        
        alpha_l = (1-config.confidence)/2
        lower_bs = int( np.round(alpha_l*config.bootstraps) )
        alpha_u = 1-alpha_l
        upper_bs = int( np.round(alpha_u*config.bootstraps) ) - 1
        
        lower_conf_limits = []
        upper_conf_limits = []
        ml_model_stderrs = []
        print('\n\tPercentile Confidence Interval ('+str(round(config.confidence*100))+'%):')
        for k in range(len(optimizer.ml_model)):
            bs_params = [bsmodel[k] for bsmodel in bsmodels]
            bs_delta = np.sort( np.array(bs_params) - optimizer.ml_model[k] )
            lower_cl = optimizer.ml_model[k] - bs_delta[upper_bs]
            if lower_cl < l1_bounds_min[k]:  lower_cl = l1_bounds_min[k]
            upper_cl = optimizer.ml_model[k] - bs_delta[lower_bs]
            if upper_cl > l1_bounds_max[k]:  upper_cl = l1_bounds_max[k]
            
            bs_stderr = np.std(bs_params,ddof=1)
            
            lower_conf_limits.append(lower_cl)
            upper_conf_limits.append(upper_cl)
            ml_model_stderrs.append(bs_stderr)
        
        
        global_metrics = pd.DataFrame({'parameter':global_params,
                                       'dw':optimizer.ml_model[:gvs],
                                       'se':ml_model_stderrs[:gvs],
                                       'lower_conf_limit':lower_conf_limits[:gvs],
                                       'upper_conf_limit':upper_conf_limits[:gvs]})
        print(global_metrics)
        
        residue_metrics = pd.DataFrame({'residue':optimizer.residues,
                                        'dw':optimizer.ml_model[gvs:],
                                        'se':ml_model_stderrs[gvs:],
                                        'lower_conf_limit':lower_conf_limits[gvs:],
                                        'upper_conf_limit':upper_conf_limits[gvs:]})
        
        optimizer.ml_model_stderrs = ml_model_stderrs
        optimizer.lower_conf_limits = lower_conf_limits
        optimizer.upper_conf_limits = upper_conf_limits
        
                                           
        
        
        
        # Compute the BCa bootstrap confidence intervals
        # Equations from http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
        # Slides 34-36
        
        ##### NEED TO FIX THE BCA CALCULATION, SOMETHING IS BROKEN #####
        '''
        print('BCa CONFIDENCE INTERVAL ('+str(round(config.confidence*100))+'%):')
        for k in range(gvs):
            frac_negative = sum([bsmodel[k] < optimizer.ml_model[k] for bsmodel in bsmodels])/len(bsmodels)
            if frac_negative == 0:
                print('Too much assymetry in bootstrap values')
                continue
            bias_corr_factor = stats.norm.ppf(frac_negative)
            
            #jackknife the acceleration factor
            jk_est = []
            for x in range(len(bsmodels)):
                param_bootstraps_i = [bsmodel[k] for bsmodel in bsmodels]
                param_bootstraps_i.pop(x)
                jk_est.append( np.std(param_bootstraps_i, ddof=1) )
            jk_est = np.array(jk_est)
            jk_mean = np.mean(jk_est)
            acc_factor = np.sum(np.power(jk_mean-jk_est,3)) / 6*np.power(np.sum(np.power(jk_mean-jk_est,2)),3/2)
            
            z_l = stats.norm.ppf((1-config.confidence)/2)
            z_u = stats.norm.ppf(1 - (1-config.confidence)/2)
            
            bca_alpha_l = stats.norm.cdf( bias_corr_factor + (bias_corr_factor + z_l)/(1 - acc_factor*(bias_corr_factor + z_l)) )
            bca_alpha_u = stats.norm.cdf( bias_corr_factor + (bias_corr_factor + z_u)/(1 - acc_factor*(bias_corr_factor + z_u)) )
            
            print((round(bca_alpha_l,2),round(bca_alpha_u,2)))
            
            lower_bs = int( np.round(bca_alpha_l*config.bootstraps) )
            upper_bs = int( np.round(bca_alpha_u*config.bootstraps) ) - 1
            
            bs_delta = np.sort( np.array(bs_params) - optimizer.ml_model[k] )
            lower_cl = optimizer.ml_model[k] - bs_delta[upper_bs]
            if lower_cl < l1_bounds_min[k]:  lower_cl = l1_bounds_min[k]
            upper_cl = optimizer.ml_model[k] - bs_delta[lower_bs]
            if upper_cl > l1_bounds_max[k]:  upper_cl = l1_bounds_max[k]
            
            print('\t'+global_params[k]+' = '+str(round(optimizer.ml_model[k],2))+' ['+str(round(lower_cl,2))+','+str(round(upper_cl,2))+']\n')
        '''
        
    
    ## Calculate the likelihood based confidence intervals
    else:   #boostraps == 0
        optimizer.mode = 'ml_optimization'
        lower_conf_limits = []
        upper_conf_limits = []
        '''
        for i in range(len(optimizer.ml_model)):
            # Decrease and increase the parameter by 0.1% until we hit the critical value of 1.92 for 95% confidence
            scaler = 1.0
            params = optimizer.ml_model
            while optimizer.fitness(params)[0] - performance['ml_model_score'] <= stats.chi2.ppf(config.confidence,df=1)/2:
                params[i] = optimizer.ml_model[i]*scaler
                scaler-=0.001
            lower_conf_limits.append(params[i])
            
            scaler = 1.0
            params = optimizer.ml_model
            while optimizer.fitness(params)[0] - performance['ml_model_score'] <= stats.chi2.ppf(config.confidence,df=1)/2:
                params[i] = optimizer.ml_model[i]*scaler
                scaler+=0.001
            upper_conf_limits.append(params[i])
        '''
        
        ## Set up a MCMC walk within the parameter space
        # Let's start with 10 random walks of 10K steps
        walks = 50
        
        # Set this up now using the mcmc function
        executor = concurrent.futures.ProcessPoolExecutor(config.num_threads)
        futures = [executor.submit(mcmc_walker, optimizer, config.mcmc_steps, config.confidence, l1_bounds_min, l1_bounds_max, config.seed, k) for k in range(config.mcmc_walks)]
        concurrent.futures.wait(futures)

        
        accepted_steps = []
        for future in futures:   accepted_steps += future.result()
        
        '''
        walk_length = 10000
        accepted_steps = []
        for i in range(walks):
            model = list(optimizer.ml_model)
            counter = 0
            consecutive_failed = 0
            for j in range(walk_length):
                prev_model = list(model)
                
                model[0]+=np.random.normal(0,0.005)
                model[1]+=np.random.normal(0,0.005)
                model[2]+=np.random.normal(0,0.1)
                model[3]+=np.random.normal(0,model[3]/100)
                model[4]+=np.random.normal(0,model[4]/100)
                model[5]+=np.random.normal(0,model[5]/100)
                model[gvs:] = list(np.array(model[gvs:]) + np.random.normal(0,0.1,len(model[gvs:])))
               
                

                # For global variables, allow it to walk randomly -/+ 0.1% and dw to move randomly -/+ 0.1 Hz
                perturbation = list((np.random.random(gvs)-0.5)/500*np.array(model[:gvs])) + list((np.random.random(len(model[gvs:]))-0.5)/5)
                model = list(np.array(model) + perturbation)
                
                #model[:gvs] = np.array(model[:gvs]) + (np.random.random(gvs)-0.5)/500*np.array(model[:gvs])
                # Allow movement of dw to be randomly -/+ 0.1 Hz
                #model[gvs:] = np.array(model[gvs:]) + (np.random.random(len(model[gvs:]))-0.5)/5
                
                if optimizer.fitness(model)[0] - performance['ml_model_score'] <= stats.chi2.ppf(config.confidence,df=1)/2:
                    accepted_steps.append(list(model))
                    counter+=1
                    consecutive_failed = 0
                else:
                    model = list(prev_model)
                    consecutive_failed += 1
                
                if consecutive_failed > 50: break
            print('Walk #'+str(i+1)+': Accepted '+str(counter)+' of '+str(j+1)+' steps')
        '''
        
        
        
        for i in range(len(optimizer.ml_model)):
            param = [x[i] for x in accepted_steps]
            #print(param)
            if len(param) > 0:
                lower_conf_limits.append(min(param))
                upper_conf_limits.append(max(param))
            else:
                lower_conf_limits.append(-1)
                upper_conf_limits.append(-1)
            
        #print(lower_conf_limits)
        
        optimizer.ml_model_stderrs = list(optimizer.ml_model) #JUST STICKING NUMBERS IN FOR NOW
        optimizer.lower_conf_limits = list(lower_conf_limits)
        optimizer.upper_conf_limits = list(upper_conf_limits)
    
    performance['phase4_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])
    
    ## Generate summary PDF
    
    print('\n---  Phase 5: Generating output files  ---\n')
    
    #Make a folder for figures
    image_dir = os.path.join(config.output_dir,'images')
    make_output_dir( image_dir )
    
    summary_pdf = CompLEx_Report(optimizer, config, performance, lam, pygmo_seed, image_dir)
    summary_pdf_name = os.path.join(config.output_dir, fname_prefix + '_CompLEx_summary.pdf')
    summary_pdf.output(summary_pdf_name)
    
    
    ## Dump out data so I can keep testing the CompLExReport function without running MaLICE
    '''
    data_dir = os.path.join(config.output_dir,'data')
    pickle.dump(optimizer, os.path.join(data_dir,'optimizer'))
    pickle.dump(config, os.path.join(data_dir,'config'))
    pickle.dump(performance, os.path.join(data_dir,'performance'))
    '''

    
    
    
    
    
    
    
    
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


    
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tFinal run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])
if __name__ == "__main__":
    main()
