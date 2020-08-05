import concurrent
import concurrent.futures
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import pygmo as pg
from scipy.optimize import minimize
import scipy.stats as stats

from malice import mcmc
from malice.args import parse_args
from malice.optimizer import MaliceOptimizer
from malice.output import create_output_files, make_output_dir
from malice.seeds import set_base_seed



def gen_pop1(optimizer):
    # Will span from 100 nM to 10 mM
    Kd_exp_random = list(np.random.random(1)*5-1)
    # Will span from 1 kHz to 10 Mhz
    kex_exp_random = list(np.random.random(1)*4+3)
    # 0 - 200 Hz
    dR2_random = list(np.random.random(1)*200)
    # random amp logic is that since amp = intensity * lw, lets just randomly
    # sample something from the reasonable intensity range and multiply by 20,
    # which is probably a decent enough guess of typical protein linewidths
    amp_scaler_random = [np.random.normal(np.mean(optimizer.data.intensity),
                         np.std(optimizer.data.intensity)) * 5]
    # 1/4 to 1/50th of mean intensity
    i_noise_random = list(np.mean(optimizer.data.intensity) /
                          (np.random.random(1)*46+4))
    # larmor / 50-4500 -- rough range of digital res
    cs_noise_random = list(optimizer.larmor/(np.random.random(1)*4450+50))
    # Every delta_w is 0-0.1 ppm CSP
    dw_random = list(0.1*optimizer.larmor *
                     np.random.random(len(optimizer.residues)))

    return Kd_exp_random + kex_exp_random + dR2_random + amp_scaler_random + \
        i_noise_random + cs_noise_random + dw_random


def gen_pop2(optimizer):
    N_random = list(np.array(optimizer.reference['15N_ref']) +
                    np.random.normal(0, optimizer.l1_model[5]/optimizer.nh_scale, len(optimizer.residues)))
    H_random = list(np.array(optimizer.reference['1H_ref']) + np.random.normal(0, optimizer.l1_model[5], len(optimizer.residues)))
    Iref_random = list(np.array(optimizer.reference['I_ref']) + np.random.normal(0, optimizer.l1_model[4], len(optimizer.residues)))

    return N_random + H_random + Iref_random


def pygmo_wrapper(optimizer, pop_generator, islands, pop_size,
                  generations, evo_rounds, tolerance):
    archi = pg.archipelago(prob=pg.problem(optimizer),
                           s_pol=pg.select_best(0.10),
                           r_pol=pg.fair_replace(0.05),
                           t=pg.fully_connected())
    archi.set_migration_type(pg.migration_type.broadcast)
    for iteration in range(islands):
        pop = pg.population(pg.problem(optimizer))
        for _ in range(pop_size):
            pop.push_back(pop_generator(optimizer))
        archi.push_back(pop=pop, algo=pg.sade(gen=generations, variant=6, variant_adptv=2, ftol=tolerance, xtol=tolerance))
    archi.evolve(evo_rounds)
    archi.wait()
    best_score = np.array(archi.get_champions_f()).min()
    best_index = archi.get_champions_f().index(best_score)
    best_model = archi.get_champions_x()[best_index]

    return best_model, best_score


def csp_trajectory(theta, data_points, nh_scale):
    return np.sum(np.square((data_points['15N']-data_points['15N_ref'])*nh_scale - data_points.csfit*np.sin(theta)) +
                  np.square((data_points['1H']-data_points['1H_ref']) - data_points.csfit*np.cos(theta)))


def main():
    args = parse_args(sys.argv[1:])
    make_output_dir(args.output_dir)
    set_base_seed(args.seed)
    run_malice(args)


def parse_input(fname):
    data = pd.read_csv(fname,
                        dtype={'residue': np.int64,
                               '15N': np.float64,
                               '1H': np.float64,
                               'intensity': np.float64,
                               'titrant': np.float64,
                               'visible': np.float64})

    residues = list(data.groupby('residue').groups.keys())
    reference_points = pd.DataFrame()
    for res in residues:
        resdata = data.copy()[data.residue == res]
        # Use the lowest titration point (hopefully zero) for the reference
        min_titrant_conc = resdata.titrant.min()
        reference_points = reference_points.append(resdata.loc[resdata.titrant == min_titrant_conc, ['residue', '15N', '1H', 'intensity']].mean(axis=0), ignore_index=True)

    return data, reference_points, residues


def run_malice(config):
    performance = {}
    performance['start_time'] = time.time()

    fname_prefix = config.input_file.split('/')[-1].split('.')[0]

    # Important variables
    larmor = config.larmor
    gvs = 6
    lam = 0.015
    nh_scale = 0.2  # Consider reducing to ~0.14

    user_data, initial_reference, residues = parse_input(config.input_file)

    i_noise_est = np.mean(user_data.intensity)/10

    optimizer = MaliceOptimizer(larmor=larmor,
                                gvs=gvs,
                                lam=lam,
                                data=user_data,
                                mode='ml_optimization',
                                cs_dist='gaussian',
                                nh_scale=nh_scale)

    # Stage 1 - parameter optimization with L1 regularization
    print('\n---  Phase 1: Differential evolution for parameter optimization with L1 regularization  ---\n')

    l1_bounds_min = [-1, 1, 0, np.min(user_data.intensity)/10, i_noise_est/10, larmor/4500] + list([0]*len(residues))
    l1_bounds_max = [4, 7, 200, np.max(user_data.intensity)*200, i_noise_est*10, larmor/5] + list([6*larmor]*len(residues))
    optimizer.set_bounds((l1_bounds_min, l1_bounds_max))

    optimizer.l1_model, performance['l1_model_score'] = pygmo_wrapper(optimizer,
                                                                      pop_generator=gen_pop1,
                                                                      islands=config.phase1_islands,
                                                                      pop_size=config.pop_size,
                                                                      generations=config.phase1_generations,
                                                                      evo_rounds=config.phase1_evo_rounds,
                                                                      tolerance=config.tolerance)
    print('\tL1-penalized -logL:\t'+str(round(performance['l1_model_score'], 2)))
    optimizer.lam = 0
    performance['l1_unpenalized_score'] = optimizer.fitness(optimizer.l1_model)[0]
    print('\tUnpenalized -logL:\t'+str(round(performance['l1_unpenalized_score'], 2)))

    print('\n\tKd = '+str(round(np.power(10.0, optimizer.l1_model[0]), 2)) +
          '\n\tkoff = '+str(round(np.power(10.0, optimizer.l1_model[1]), 2)) +
          '\n\tdR2 = '+str(round(optimizer.l1_model[2], 2)) +
          '\n\tAmp_scaler = '+str(round(optimizer.l1_model[3], 2)) +
          '\n\tInoise = '+str(round(optimizer.l1_model[4], 2)) +
          '\n\tCSnoise = '+str(round(optimizer.l1_model[5], 2)) +
          '\n\tMax dw = '+str(round(np.max(optimizer.l1_model[gvs:]), 2)))

    performance['phase1_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])

    # Stage 2 - reference peak optimization
    print('\n---  Phase 2: Reference peak optimization  ---\n')

    optimizer.mode = 'reference_optimization'
    optimizer.cs_dist = 'rayleigh'

    ref_opt_bounds_min = list(np.array(initial_reference['15N']) - 0.2) + list(
                               np.array(initial_reference['1H']) - 0.05) + list(np.array(initial_reference.intensity)/10)
    ref_opt_bounds_max = list(np.array(initial_reference['15N']) + 0.2) + list(
                               np.array(initial_reference['1H']) + 0.05) + list(np.array(initial_reference.intensity)*10)
    optimizer.set_bounds((ref_opt_bounds_min, ref_opt_bounds_max))

    pre_ref = optimizer.reference.copy()

    opt_ref, performance['opt_ref_score'] = pygmo_wrapper(optimizer,
                                                          pop_generator=gen_pop2,
                                                          islands=config.phase2_islands,
                                                          pop_size=config.pop_size,
                                                          generations=config.phase2_generations,
                                                          evo_rounds=config.phase2_evo_rounds,
                                                          tolerance=config.tolerance)
    optimizer.reference = pd.DataFrame({'residue': residues,
                                        '15N_ref': opt_ref[:int(len(opt_ref)/3)],
                                        '1H_ref': opt_ref[int(len(opt_ref)/3):int(2*len(opt_ref)/3)],
                                        'I_ref': opt_ref[int(2*len(opt_ref)/3):]})

    print('\t-logL after reference optimization:\t'+str(round(performance['opt_ref_score'], 2)))

    # Output the optimized references as a csv
    reference_delta_df = pre_ref.copy().rename(columns={'15N_ref': '15N_original',
                                                        '1H_ref': '1H_original',
                                                        'I_ref': 'Intensity_original'})
    reference_delta_df['15N_optimized'] = opt_ref[:int(len(opt_ref)/3)]
    reference_delta_df['1H_optimized'] = opt_ref[int(len(opt_ref)/3):int(2*len(opt_ref)/3)]
    reference_delta_df['Intensity_optimized'] = opt_ref[int(2*len(opt_ref)/3):]
    reference_delta_df['15N_delta'] = reference_delta_df['15N_original'] - reference_delta_df['15N_optimized']
    reference_delta_df['1H_delta'] = reference_delta_df['1H_original'] - reference_delta_df['1H_optimized']
    reference_delta_df['Intensity_delta'] = reference_delta_df['Intensity_original'] - reference_delta_df['Intensity_optimized']
    reference_delta_df = reference_delta_df[['residue', '15N_original', '15N_optimized', '15N_delta',
                                             '1H_original', '1H_optimized', '1H_delta',
                                             'Intensity_original', 'Intensity_optimized', 'Intensity_delta']]
    reference_delta_df = reference_delta_df.round(3)
    reference_csv = os.path.join(config.output_dir, fname_prefix + '_reference_peaks.csv')
    reference_delta_df.to_csv(reference_csv, index=False)

    print('\tSaved reference optimization results to '+reference_csv)

    performance['phase2_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])

    # Stage 3 - polish off the model with gradient minimization
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
        if scale_opt_bounds_min[i] < l1_bounds_min[i]:
            scale_opt_bounds_min[i] = l1_bounds_min[i]
        if scale_opt_bounds_max[i] > l1_bounds_max[i]:
            scale_opt_bounds_max[i] = l1_bounds_max[i]
    optimizer.set_bounds((scale_opt_bounds_min, scale_opt_bounds_max))

    scaled_opt_initial = list(optimizer.l1_model[:gvs]) + [1]

    scaled_dw_opt = minimize(optimizer.fitness, scaled_opt_initial, method='SLSQP', bounds=optimizer.get_scipy_bounds(),
                             tol=1e-7, options={'disp': False,
                                                'maxiter': config.least_squares_max_iter,
                                                'ftol': 1e-7})

    performance['scaled_dw_score'] = scaled_dw_opt.fun
    dw_scaler = scaled_dw_opt.x[-1]
    print('\n\tOptimized delta_w scaler: '+str(round(dw_scaler, 2)) +
          '\n\tPost-scalar optimization -logL: '+str(round(performance['scaled_dw_score'], 2)))

    # Run the fine tuning optimization
    optimizer.mode = 'ml_optimization'

    # Bounds
    ml_opt_bounds_min = [scaled_dw_opt.x[0]-0.2, scaled_dw_opt.x[1]-0.2, scaled_dw_opt.x[2]-10, scaled_dw_opt.x[3]/1.2,
                         scaled_dw_opt.x[4]/1.2, scaled_dw_opt.x[5]/1.2] + [0]*len(residues)
    ml_opt_bounds_max = [scaled_dw_opt.x[0]+0.2, scaled_dw_opt.x[1]+0.2, scaled_dw_opt.x[2]+10, scaled_dw_opt.x[3]*1.2,
                         scaled_dw_opt.x[4]*1.2, scaled_dw_opt.x[5]*1.2] + list(np.array(optimizer.l1_model[gvs:])*scaled_dw_opt.x[-1]*2)
    # Fix any of the global bounds that go off into stupid places
    for i in range(gvs):
        if ml_opt_bounds_min[i] < l1_bounds_min[i]:
            ml_opt_bounds_min[i] = l1_bounds_min[i]
        if ml_opt_bounds_max[i] > l1_bounds_max[i]:
            ml_opt_bounds_max[i] = l1_bounds_max[i]
    optimizer.set_bounds((ml_opt_bounds_min, ml_opt_bounds_max))

    ml_opt_initial = list(scaled_dw_opt.x[:gvs]) + list(np.array(optimizer.l1_model[gvs:])*dw_scaler)

    # Full minimization
    ml_model_opt = minimize(optimizer.fitness, ml_opt_initial, method='SLSQP',
                            bounds=optimizer.get_scipy_bounds(), tol=1e-7,
                            options={'disp': False,
                                     'maxiter': config.least_squares_max_iter,
                                     'ftol': 1e-7})

    performance['ml_model_score'] = ml_model_opt.fun
    print('\n\tFinal -logL: '+str(round(performance['ml_model_score'], 2)))

    optimizer.ml_model = ml_model_opt.x

    # Let's go ahead and print some results
    print('\n\tKd = '+str(round(np.power(10.0, optimizer.ml_model[0]), 2)) +
          '\n\tkoff = '+str(round(np.power(10.0, optimizer.ml_model[1]), 2)) +
          '\n\tdR2 = '+str(round(optimizer.ml_model[2], 2)) +
          '\n\tAmp_scaler = '+str(round(optimizer.ml_model[3], 2)) +
          '\n\tInoise = '+str(round(optimizer.ml_model[4], 2)) +
          '\n\tCSnoise = '+str(round(optimizer.ml_model[5], 2)) +
          '\n\tMax dw = '+str(round(np.max(optimizer.ml_model[gvs:]), 2)))

    performance['phase3_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])

    # Stage 4 - MCMC walk in parameter space to estimate confidence intervals
    print('\n---  Phase 4: Confidence interval estimation of global parameters and delta w  ---\n')
    print('\tNumber of MCMC walks: '+str(config.mcmc_walks))

    optimizer.mode = 'ml_optimization'
    

    executor = concurrent.futures.ProcessPoolExecutor(config.num_threads)
    futures = [executor.submit(mcmc.walk, optimizer, config.mcmc_steps, config.confidence, l1_bounds_min, l1_bounds_max, k) for k in range(config.mcmc_walks)]
    concurrent.futures.wait(futures)

    accepted_steps = []
    for future in futures:
        accepted_steps = accepted_steps + future.result() 
    # Sort the MCMC runs by logL
    #accepted_steps.sort(key=lambda x:optimizer.fitness(x)[0])
    accepted_step_logLs = [optimizer.fitness(m)[0] for m in accepted_steps]
    # Do a fast sorting on the models
    sorted_steps = [x for _,x in sorted(zip(accepted_step_logLs, accepted_steps))]
    # Temporary output file so I can QC how spread out the logL scores are, may add as a real output
    #optimizer.mcmc_logL = accepted_stop_logLs
    fout = open('logL.txt','w')
    for logL in accepted_step_logLs:
        fout.write(format(logL,'.3f')+'\n')
    fout.close()


    ## Set up initial lower/upper confidence levels as the ML model
    confidences = []
    lower_conf_values = list(optimizer.ml_model)
    upper_conf_values = list(optimizer.ml_model)
    for model in sorted_steps:
        # Since the models are in order of increasing logL, compute the %ile that the model represents, and update the quantiles
        # as needed for the min/max values
        delta_logL = optimizer.fitness(model)[0] - performance['ml_model_score']
        model_conf_level = stats.chi2.cdf( 2*delta_logL, df=1)
        lower_quantile = (1 - model_conf_level)/2
        upper_quantile = 1 - lower_quantile

        for k in range(len(optimizer.ml_model)):
            if model[k] < lower_conf_values[k]: lower_conf_values[k] = model[k]
            if model[k] > upper_conf_values[k]: upper_conf_values[k] = model[k]
        
        confidences.append( [lower_quantile] + lower_conf_values )
        confidences.append( [upper_quantile] + upper_conf_values )


    '''
    ## Generate increments of the confidence interval to estimate confidence values and their matched chi2 densities
    conf_values = np.linspace(0.001, config.confidence, 990)  # if using default 0.99, will create points on every 0.1%ile
    #lower_conf_level_sets = []
    #upper_conf_level_sets = []
    confidences = []
    for conf_value in list(conf_values):

        # Identity the set of models with logL <= max_logL + chi2.ppf(conf_level)/2
        threshold_logL = performance['ml_model_score'] + stats.chi2.ppf(1-conf_value, df=1)/2
        target_index = np.argmin( np.abs(np.array(accepted_step_logLs) - threshold_logL) )
        focal_steps = accepted_steps[:target_index+1]

        lower_conf_levels = []
        upper_conf_levels = []
        for p in range(len(optimizer.ml_model)):
            param = [x[p] for x in focal_steps]
            if len(param) > 0:
                lower_conf_levels.append(min(param))
                upper_conf_levels.append(max(param))
            else:
                lower_conf_levels.append(optimizer.ml_model[p])
                upper_conf_levels.append(optimizer.ml_model[p])
        
        lower_level = (1-conf_value)/2
        upper_level = 1 - lower_level

        confidences.append( [lower_level]+lower_conf_levels )
        confidences.append( [upper_level]+lower_conf_levels )
        #confidence_df[lower_level] = lower_conf_levels
        #confidence_df[upper_level] = upper_conf_levels

        #lower_conf_level_sets.append( lower_conf_levels )
        #upper_conf_level_sets.append( upper_conf_levels )
    '''

    ## For the time being, let's just have it spit out the confidence_df as a table, and set the optimizer values to
    ## the 95%ile so that it functions similarly. Will give me a chance to do some quality control checks now...
    
    confidence_df = pd.DataFrame( confidences, columns=['conf_level']+list(range(len(optimizer.ml_model))) )
    confidence_df = confidence_df.sort_values('conf_level')
    optimizer.confidence_df = confidence_df
    
    
    optimizer.lower_conf_limits = list( confidence_df.loc[confidence_df.conf_level.sub(0.025).abs().idxmin(), 
                                                          confidence_df.columns[1:]] )
    optimizer.upper_conf_limits = list( confidence_df.loc[confidence_df.conf_level.sub(0.975).abs().idxmin(), 
                                                          confidence_df.columns[1:]] )

    performance['phase4_time'] = time.time() - performance['start_time']
    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tCurrent run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])

    # Infer CSP trajectories
    print('\n---  Phase 5: CSP trajectory inference  ---\n')

    fit_points = optimizer.pfitter()

    thetas = []
    theta_F_stats = []
    theta_p_values = []
    for resi in residues:
        residue_fits = fit_points.copy()[fit_points.residue == resi]
        theta_est = minimize(csp_trajectory, np.pi/4, args=(residue_fits, nh_scale), method='L-BFGS-B').x[0]
        theta_sumsq = csp_trajectory(theta_est, residue_fits, nh_scale)

        theta_F_stat = ((np.sum(np.square((residue_fits['15N'] - residue_fits['15N_ref'])*nh_scale) +
                                np.square(residue_fits['1H'] - residue_fits['1H_ref'])) - theta_sumsq)/2)/(
                         theta_sumsq/(len(residue_fits)-4))

        theta_p_value = 1 - stats.f.cdf(theta_F_stat, 2, len(residue_fits)-4)

        thetas.append(theta_est)
        theta_F_stats.append(theta_F_stat)
        theta_p_values.append(theta_p_value)

    optimizer.thetas = thetas
    optimizer.theta_F = theta_F_stats
    optimizer.theta_up = theta_p_values

    print('\n---  Phase 6: Output file generation  ---\n')

    create_output_files(optimizer, config.confidence, gvs, residues, fname_prefix,
                        config.output_dir, config, lam,  performance, fit_points)

    performance['current_time'] = time.time() - performance['start_time']
    print('\n\tFinal run time = '+str(datetime.timedelta(seconds=performance['current_time'])).split('.')[0])


if __name__ == "__main__":
    main()
