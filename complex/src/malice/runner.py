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
    parser.add_argument('--least_squares_max_iter', #TODO: Different maxes for different phases?
                        type=int,
                        help='Maximum number of iterations to run sequential least squares minimization',
                        default=100000)
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
    #TODO: find a better place for this
    parser.add_argument('--s3_prefix',
                        type=str,
                        help='S3 bucket and key prefix to upload zip to. Do not include trailing "/"',
                        default=None)
    # TODO: validate arguments
    return parser.parse_args()

def gen_pop1(optimizer):
    Kd_exp_random = list(np.random.random(1)*5-1)   # Will span from 100 nM to 10 mM
    kex_exp_random = list(np.random.random(1)*4+3)  # Will span from 1 kHz to 10 Mhz
    dR2_random = list(np.random.random(1)*200)            # 0 - 200 Hz
    amp_scaler_random = [np.random.normal(np.mean(optimizer.data.intensity),np.std(optimizer.data.intensity)) * 5]
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

def csp_trajectory(theta, data_points, nh_scale):
    return np.sum(  np.square( (data_points['15N']-data_points['15N_ref'])*nh_scale - data_points.csfit*np.sin(theta) ) +
                    np.square( (data_points['1H']-data_points['1H_ref']) - data_points.csfit*np.cos(theta) )   )

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
    data = input.copy()[input.intensity > 0]
    
    residues = list(data.groupby('residue').groups.keys())
    reference_points = pd.DataFrame()
    for res in residues:
        resdata = data.copy()[data.residue == res]
        ## Use the lowest titration point (hopefully zero) for the reference
        min_titrant_conc = resdata.titrant.min()
        reference_points = reference_points.append(resdata.loc[resdata.titrant == min_titrant_conc,['residue','15N','1H','intensity']].mean(axis=0),ignore_index=True)
        
        #if len(list(resdata.titrant[resdata.titrant == min_titrant_conc])) == 1:
        #    reference_points = reference_points.append(resdata.loc[resdata.titrant == min_titrant_conc,['residue','15N','1H','intensity']])
        #else:
            
    
    return data, reference_points, residues

def run_malice(config):
    performance = {}
    performance['start_time'] = time.time()
    
    fname_prefix = config.input_file.split('/')[-1].split('.')[0]
    
    ## Important variables
    larmor = config.larmor
    gvs = 6
    lam = 0.015
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
    l1_bounds_max = [4, 7, 200, np.max(user_data.intensity)*200, i_noise_est*10, larmor/5] + list([6*larmor]*len(residues))
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
    
    print('\n\tKd = '+str(round(np.power(10.0,optimizer.l1_model[0]),2))+
          '\n\tkoff = '+str(round(np.power(10.0,optimizer.l1_model[1]),2))+
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
    print('\n\tKd = '+str(round(np.power(10.0,optimizer.ml_model[0]),2))+
          '\n\tkoff = '+str(round(np.power(10.0,optimizer.ml_model[1]),2))+
          '\n\tdR2 = '+str(round(optimizer.ml_model[2],2))+
          '\n\tAmp_scaler = '+str(round(optimizer.ml_model[3],2))+
          '\n\tInoise = '+str(round(optimizer.ml_model[4],2))+
          '\n\tCSnoise = '+str(round(optimizer.ml_model[5],2))+
          '\n\tMax dw = '+str(round(np.max(optimizer.ml_model[gvs:]),2)))
    
    performance['phase3_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])
    
    ## Stage 4 - MCMC walk in parameter space to estimate confidence intervals
    print('\n---  Phase 4: Confidence interval estimation of global parameters and delta w  ---\n')
    print('\tNumber of MCMC walks: '+str(config.mcmc_walks))
    
    optimizer.mode = 'ml_optimization'
    lower_conf_limits = []
    upper_conf_limits = []
    
    executor = concurrent.futures.ProcessPoolExecutor(config.num_threads)
    futures = [executor.submit(mcmc_walker, optimizer, config.mcmc_steps, config.confidence, l1_bounds_min, l1_bounds_max, config.seed, k) for k in range(config.mcmc_walks)]
    concurrent.futures.wait(futures)
    
    accepted_steps = []
    for future in futures:   accepted_steps += future.result()
    
    for i in range(len(optimizer.ml_model)):
        param = [x[i] for x in accepted_steps]
        #print(param)
        if len(param) > 0:
            lower_conf_limits.append(min(param))
            upper_conf_limits.append(max(param))
        else:
            lower_conf_limits.append(-1)
            upper_conf_limits.append(-1)
    
    optimizer.lower_conf_limits = list(lower_conf_limits)
    optimizer.upper_conf_limits = list(upper_conf_limits)
    
    performance['phase4_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])
    
    ## Infer CSP trajectories
    print('\n---  Phase 5: CSP trajectory inference  ---\n')

    optimizer.mode = 'pfitter'
    fit_points = optimizer.fitness()

    thetas = []
    theta_F_stats = []
    theta_p_values = []
    for resi in residues:
        residue_fits = fit_points.copy()[fit_points.residue == resi]
        theta_est = minimize( csp_trajectory, np.pi/4, args=(residue_fits, nh_scale), method='L-BFGS-B' ).x[0]
        theta_sumsq = csp_trajectory(theta_est,residue_fits,nh_scale)

        theta_F_stat = ( ( np.sum( np.square( (residue_fits['15N'] - residue_fits['15N_ref'])*nh_scale ) + 
                                   np.square( residue_fits['1H'] - residue_fits['1H_ref'] ) ) - theta_sumsq )/2 )/(
                         theta_sumsq/(len(residue_fits)-4) )

        theta_p_value = 1 - stats.f.cdf(theta_F_stat, 2, len(residue_fits)-4)

        thetas.append(theta_est)
        theta_F_stats.append(theta_F_stat)
        theta_p_values.append(theta_p_value)
    
    optimizer.thetas = thetas
    optimizer.theta_F = theta_F_stats
    optimizer.theta_up = theta_p_values
    
    
    
    print('\n---  Phase 6: Output file generation  ---\n')
    
    ## Generate a CSV with confidence intervals for all of the delta_w's and trajectories if available
    alpha = 100.0*(1-config.confidence)/2
    deltaw_df = pd.DataFrame({'residue':residues,'delta_w':optimizer.ml_model[gvs:]/larmor,
                              'conf_limit_'+str(round(alpha,1)):np.array(optimizer.lower_conf_limits[gvs:])/larmor,
                              'conf_limit_'+str(round(100-alpha,1)):np.array(optimizer.upper_conf_limits[gvs:])/larmor,
                              'estimated_theta':optimizer.thetas, 'theta_F_stat':optimizer.theta_F,
                              'uncorrected_theta_p_value':optimizer.theta_up})
    deltaw_df['corrected_theta_p_value'] = deltaw_df.uncorrected_theta_p_value * len(residues) / deltaw_df.uncorrected_theta_p_value.rank(ascending=False)
    optimizer.deltaw_df = deltaw_df
    
    ## Print out data
    csv_name = os.path.join(config.output_dir, fname_prefix + '_CompLEx_deltaw.csv')
    txt_name = os.path.join(config.output_dir, fname_prefix+'_CompLEx_deltaw.txt')
    deltaw_df.to_csv(csv_name,index=False)
    deltaw_df[['residue','delta_w']].to_csv(txt_name, index=False,header=False)
    
    ## Record the fit points to a file
    fit_points_name = os.path.join(config.output_dir, fname_prefix+'_CompLEx_fit_points.csv')
    fit_points.to_csv(fit_points_name, index=False)
    
    #Make a folder for figures
    image_dir = os.path.join(config.output_dir,'images')
    make_output_dir( image_dir )
    
    ## Generate summary PDF
    summary_pdf = CompLEx_Report(optimizer, config, performance, lam, pygmo_seed, image_dir)
    summary_pdf_name = os.path.join(config.output_dir, fname_prefix + '_CompLEx_summary.pdf')
    summary_pdf.output(summary_pdf_name)
    
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tFinal run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])
    
if __name__ == "__main__":
    main()
