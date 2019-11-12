### MaLICE v190923-0903
### First standalone script version

## Notable features:    - Split the global variable and delta_w optimization
##                          from reference peak optimization
##                        - Currently using additive C
##                            - Need to go back validate this is best

## Import libraries
import sys, itertools, time, datetime, concurrent, multiprocessing
import os
import numpy as np, pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize,basinhopping,differential_evolution
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import concurrent.futures

from malice.optimizer import MaliceOptimizer

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', 
                        type=str, 
                        help='path to the CSV to import')
    parser.add_argument("--pop_iter", 
                        type=int,
                        help='Number of populations to perform the differential evolution on',
                        default=10)
    parser.add_argument('--pop_size', 
                        type=int,
                        help='Number of populations to perform differential evolution on',
                        default=20)
    parser.add_argument('--evo_max_iter',  #TODO: Different maxes for different phases?
                        type=int,
                        help='Maximum number of iterations to run differential evolution for each population',
                        default=100000)
    parser.add_argument('--least_squares_max_iter', #TODO: Different maxes for different phases?
                        type=int,
                        help='Maximum number of iterations to run sequential least squares minimization',
                        default=10000)
    parser.add_argument('--thread_count',
                        type=int,
                        help='Number of threads to spawn',
                        default=3)
    parser.add_argument('--bootstraps',
                        type=int,
                        help='Number of bootstraps to perform',
                        default=100)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to store output files. Creates if non-existent.',
                        default="output")
    parser.add_argument('--deterministic',
                        action='store_true',
                        help='Whether to use a deterministic seed.')
    parser.add_argument('--larmor',
                        type=int,
                        help="Larmor frequency (MHz) of 1H in the given magnetic field.",
                        default=500)
    #TODO: find a better place for this
    parser.add_argument('--s3_prefix',
                        type=str,
                        help='S3 bucket and key prefix to upload zip to. Do not include trailing "/"',
                        default=None)
    # TODO: validate arguments
    return parser.parse_args()

def model_fitter(df, model3b):
    Kd_exp = model3b.x[0]
    koff_exp = model3b.x[1]
    C = model3b.x[2]
    i_noise = model3b.x[3]
    cs_noise = model3b.x[4]
    amp = model3b.x[5]
    
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
    
    #Calculate intensity likelihood
    i_broad = pa*pb*np.square(df.dw)*kex * (np.square(kex)+(1-5*pa*pb)*np.square(df.dw))/broad_denom
    #ihat = df.I_ref/(pa + pb*C + i_broad/lw)
    df['ihat'] = df.I_ref/( pa + pb + df.I_ref*(pb*C + i_broad)/amp )
    
    #Calculate cs likelihood
    cs_broad = pa*pb*(pa-pb)*np.power(df.dw,3) * (np.square(kex)+(1-3*pa*pb)*np.square(df.dw))/broad_denom
    df['cshat'] = pb*df.dw - cs_broad
    
    return df

def null_calculator(fx, config, mleinput, model, df, gvs, bds, res):
    dfx = df.copy()
    dfx.loc[dfx.residue == res,'dw'] = 0
    
    mininit = [model[0]-0.2, model[1]-0.2, model[2]-10, model[3]/1.2,
               model[4]/1.2, model[5]/1.2, model[6]/1.2] + list(dfx.dw/2)
    maxinit = [model[0]+0.2, model[1]+0.2, model[2]+10, model[3]*1.2,
               model[4]*1.2, model[5]*1.2, model[6]*1.2] + list(dfx.dw*2)
    
    for i in range(gvs):
        if mininit[i] < bds[i][0]:	mininit[i] = bds[i][0]
        if maxinit[i] > bds[i][1]:	maxinit[i] = bds[i][1]
    
    bdsx = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])
    
    initx = list(model)
    
    # Minimize
    nullLL = minimize(fx, initx, args=(mleinput,), method='SLSQP', bounds=bdsx,
                      tol=1e-7, options={'disp':True,'maxiter':config.least_squares_max_iter})

    print('null LogL calculated for residue '+str(res))
    
    return nullLL.fun

def bootstrapper(fx, config, mleinput, model, gvs, bds):
    mleinput = mleinput.sample(frac=1,replace=True)
    
    mininit = [model[0]-0.2, model[1]-0.2, model[2]-10, model[3]/1.2,
               model[4]/1.2, model[5]/1.2, model[6]/1.2] + list(model[gvs:]/2)
    maxinit = [model[0]+0.2, model[1]+0.2, model[2]+10, model[3]*1.2,
               model[4]*1.2, model[5]*1.2, model[6]*1.2] + list(model[gvs:]*2)
    
    for i in range(gvs):
        if mininit[i] < bds[i][0]:	mininit[i] = bds[i][0]
        if maxinit[i] > bds[i][1]:	maxinit[i] = bds[i][1]
    
    bdsx = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])
    
    initx = list(model)
    
    ## Run the minimizer
    bootstrap = minimize(fx, initx, args=(mleinput,), method='SLSQP', bounds=bdsx,
                         tol=1e-7, options={'disp':False,'maxiter':config.least_squares_max_iter})
    
    return bootstrap

def main():
    args = _parse_args()
    make_output_dir(args.output_dir)
    if args.deterministic:
        np.random.seed(seed=1337)
    run_malice(args)
    
def make_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def parse_input(fname, larmor, nh_scale):
    input = pd.read_csv(fname)
    mleinput = input.copy()
    
    residues = list(mleinput.groupby('residue').groups.keys())
    resgrouped = pd.DataFrame()
    for res in residues:
        resdata = mleinput.copy()[mleinput.residue == res]
        resgrouped = resgrouped.append(resdata.loc[resdata.intensity == np.max(resdata.intensity),['residue','15N','1H','intensity']])
        #resgrouped['15N'] = [x + np.random.normal()/larmor/nh_scale for x in list(resgrouped['15N'])]
        #resgrouped['1H'] = [x + np.random.normal()/larmor for x in list(resgrouped['1H'])]
        #resgrouped['intensity'] = [x + np.random.normal()*np.mean(mleinput.intensity)/100 for x in list(resgrouped['intensity'])]
    
    return mleinput, resgrouped, residues

def run_malice(config):
    starttime = time.time()
    
    ## Important variables
    larmor = config.larmor
    gvs = 7
    lam = 0.01
    nh_scale = 0.2  # Start here, update after optimized in phase 1
    bootstraps = config.bootstraps
    
    mleinput, resgrouped, residues = parse_input(config.input_file, larmor, nh_scale)
    
    i_noise_est = np.mean(mleinput.intensity)/10
    
    optimizer = MaliceOptimizer(larmor=larmor, 
                                gvs=gvs, 
                                lam=lam,  
                                resgrouped=resgrouped,
                                residues=residues, 
                                mode='global+dw',
                                cs_dist='gaussian')

    ## Perform pop_iter replicates of pop_size member populations
    print('\n---  Round 1: initial global variable and delta w optimization  ---\n')
    models1 = []
    for iteration in range(config.pop_iter):
        ## Let's define a new starting population every time to try to introduce more coverage of the space
        pop = []
        for x in range(config.pop_size):
            Kd_exp_random = list(np.random.random(1)*5-1)   # Will span from 100 nM to 10 mM
            kex_exp_random = list(np.random.random(1)*4+3)  # Will span from 1 kHz to 10 Mhz
            dR2_random = list(np.random.random(1)*200)            # 0 - 200 Hz
            amp_random = [np.random.normal(np.mean(mleinput.intensity),np.std(mleinput.intensity)) * 20]
                # random amp logic is that since amp = intensity * lw, lets just randomly sample something from the reasonable intensity
                # range and multiply by 20, which is probably a decent enough guess of typical protein linewidths
            nh_scale_random = list(np.random.random(1)*0.25+0.05)   # Random scalar between 0.05 and 0.3
            i_noise_random = list(np.mean(mleinput.intensity)/(np.random.random(1)*46+4)) # 1/4 to 1/50th of mean intensity
            cs_noise_random = list(larmor/(np.random.random(1)*4450+50)) # larmor / 50-4500 -- rough range of digital res

            dw_random = list( 0.1*larmor * np.random.random(len(residues)) ) ## Every delta_w is 0-0.1 ppm CSP

            pop.append(Kd_exp_random + kex_exp_random + dR2_random + amp_random + nh_scale_random + i_noise_random + cs_noise_random + dw_random)


        mininit = [-1, 1, 0, np.min(mleinput.intensity)/10, 0.05, i_noise_est/10, larmor/4500] + list([0]*len(residues))
        maxinit = [4, 7, 200, np.max(mleinput.intensity)*200, 0.3, i_noise_est*10, larmor/50] + list([6*larmor]*len(residues))

        bds1 = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])
        
        ## Run the job
        print('Round 1 - Population # '+str(iteration+1))
        optimizer.i = 0
        initfit = differential_evolution(optimizer.mle, bds1, args=(mleinput,), init = pop, updating='deferred', 
                                         workers = config.thread_count, mutation=(0.5,1.9),maxiter = config.evo_max_iter, 
                                         strategy = 'best1bin', polish = False, recombination = 0.7, 
                                         tol=1e-6, disp=False, callback=optimizer.counter_factory(mleinput))
        print('\n'+str(round(initfit.fun - lam*np.sum(initfit.x[gvs:]),2))+'\n')
        models1.append(initfit)

    ## Sort the results by -logL and use the best one for reference peak optimization
    models1.sort(key=lambda y:y.fun - lam*np.sum(y.x[gvs:]))
    model1 = models1[0]
    
    optimizer.model1 = model1.x
    optimizer.residues = residues
    optimizer.mode = 'refpeak_opt'

    ## Stage 2 - reference peak optimization
    print('\n---  Round 2: reference peak optimization  ---\n')
    
    ## Set up replicate populations and run
    models2 = []
    for z in range(config.pop_iter):
        pop = []
        for x in range(config.pop_size):
            N_random = list( np.array(resgrouped['15N']) + np.random.normal(0,model1.x[6]/model1.x[4],len(residues)) )
            H_random = list( np.array(resgrouped['1H']) + np.random.normal(0,model1.x[6],len(residues)) )  
            Iref_random = list( np.array(resgrouped['intensity']) + np.random.normal(0,model1.x[3],len(residues)) )

            pop.append(N_random + H_random + Iref_random)
        pop.append( list(resgrouped['15N']) + list(resgrouped['1H']) + list(resgrouped['intensity']) )

        mininit =  list(np.array(resgrouped['15N']) - 0.2) + list(
                        np.array(resgrouped['1H']) - 0.05) + list(np.array(resgrouped.intensity)/10) 
        maxinit =  list(np.array(resgrouped['15N']) + 0.2) + list(
                        np.array(resgrouped['1H']) + 0.05) + list(np.array(resgrouped.intensity)*10) 

        bds2 = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])
        
        print('Round 2 - Population #'+str(z+1))
        optimizer.i = 0
        ref_opt = differential_evolution(optimizer.mle, bds2, args=(mleinput,), init = pop, updating='deferred', 
                                         workers = config.thread_count, mutation=(0.5,1.9), maxiter = config.evo_max_iter, 
                                         strategy = 'best1bin', polish = False, recombination = 0.7,
                                         tol=1e-7, disp=False, callback=optimizer.counter_factory(mleinput))
        print('\n'+str(round(ref_opt.fun,2))+'\n')
        models2.append(ref_opt)

    ## Sort the results by -logL and use the best one for final polishing
    models2.sort(key=lambda y:y.fun)
    model2 = models2[0]

    ## Polish the reference peaks
    model2opt = minimize(optimizer.mle, model2.x, args=(mleinput,), method='SLSQP',bounds=bds2,
                         tol=1e-7, options={'disp':True,'maxiter': config.least_squares_max_iter})

    ## Stage 3 - polish off the model with gradient minimization
    print('\n---  Round 3: gradient minimization of global variables and delta w  ---\n')
    
    ## Run the 3a scaling optimization
    # Bounds
    mininit = [model1.x[0]-1, model1.x[1]-1, model1.x[2]-20, model1.x[3]/4, 
               model1.x[4]/1.5, model1.x[5]/4, model1.x[6]/4, 0.1]
    maxinit = [model1.x[0]+1, model1.x[1]+1, model1.x[2]+20, model1.x[3]*4, 
               model1.x[4]*1.5, model1.x[5]*4, model1.x[6]*4, 10]

    # Fix any of the global bounds that go off into stupid places
    for i in range(gvs):
        if mininit[i] < bds1[i][0]:    mininit[i] = bds1[i][0]
        if maxinit[i] > bds1[i][1]:    maxinit[i] = bds1[i][1]

    bds3a = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])

    init3a = list(model1.x[:gvs]) + [1]
    
    optimizer.model2 = model2opt.x
    optimizer.lam=0
    optimizer.mode = 'dw_scale'

    model3a = minimize(optimizer.mle, init3a, args=(mleinput,), method='SLSQP',bounds=bds3a,
                       tol=1e-7,options={'disp':True,'maxiter':config.least_squares_max_iter})


    ## Run the 3b fine tuning optimization
    # Bounds
    mininit = [model3a.x[0]-0.2, model3a.x[1]-0.2, model3a.x[2]-10, model3a.x[3]/1.2,
               model3a.x[4]/1.2, model3a.x[5]/1.2, model3a.x[6]/1.2] + [0]*len(residues)
    maxinit = [model3a.x[0]+0.2, model3a.x[1]+0.2, model3a.x[2]+10, model3a.x[3]*1.2,
               model3a.x[4]*1.2, model3a.x[5]*1.2, model3a.x[6]*1.2] + list(np.array(model1.x[gvs:])*model3a.x[-1]*2)

    # Fix any of the global bounds that go off into stupid places
    for i in range(gvs):
        if mininit[i] < bds1[i][0]:    mininit[i] = bds1[i][0]
        if maxinit[i] > bds1[i][1]:    maxinit[i] = bds1[i][1]

    bds3b = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])

    init3b = list(model3a.x[:gvs]) + list(np.array(model1.x[gvs:])*model3a.x[-1])
    
    optimizer.mode = 'final_opt'

    # Full minimization
    model3b = minimize(optimizer.mle, init3b, args=(mleinput,), method='SLSQP', bounds=bds3b,
                      tol=1e-7, options={'disp':True,'maxiter':config.least_squares_max_iter})

    print('\nFinal Score = '+str(round(model3b.fun,2)))


    ## Let's go ahead and print some results + save a figure
    print('\n\tKd = '+str(round(np.power(10,model3b.x[0]),2))+
          '\n\tkoff = '+str(round(np.power(10,model3b.x[1]),2))+
          '\n\tdR2 = '+str(round(model3b.x[2],2))+
          '\n\tAmp = '+str(round(model3b.x[3],2))+
          '\n\tnh_scale = '+str(round(model3b.x[4],2))+
          '\n\tInoise = '+str(round(model3b.x[5],2))+
          '\n\tCSnoise = '+str(round(model3b.x[6],2)))

    dfs = pd.DataFrame({'residue':residues,'dw':model3b.x[gvs:]})

    ## Output the per-residue fits
    concs = mleinput[['tit','obs']].drop_duplicates()
    tit_obs_lm = stats.linregress(concs.tit,concs.obs)

    tit_rng = np.max(concs.tit) - np.min(concs.tit)
    tit_vals = np.linspace(np.min(concs.tit)-0.1*tit_rng,
                           np.max(concs.tit)+0.1*tit_rng,
                           1000)
    obs_vals = tit_obs_lm.slope*tit_vals + tit_obs_lm.intercept

    fitter_input = pd.DataFrame({'tit':tit_vals,'obs':obs_vals})
    res_df = pd.DataFrame(itertools.product(tit_vals,residues),
                          columns=['tit','residue'])
    fitter_input = pd.merge(fitter_input,res_df,on='tit')

    resparams = pd.DataFrame({'residue':residues,
                              '15N_ref':model2opt.x[:int(len(model2opt.x)/3)], 
                              '1H_ref':model2opt.x[int(len(model2opt.x)/3):2*int(len(model2opt.x)/3)],
                              'I_ref':model2opt.x[2*int(len(model2opt.x)/3):],
                              'dw':model3b.x[gvs:]})
    fitter_input = pd.merge(fitter_input,resparams,on='residue')
    
    fit_data = model_fitter(fitter_input, model3b)
    mleoutput = pd.merge(mleinput,resparams,on='residue')
    mleoutput['csp'] = larmor*(
                        np.sqrt( 
                            np.square(model3b.x[4]*(mleoutput['15N'] - mleoutput['15N_ref'])) + 
                            np.square(mleoutput['1H'] - mleoutput['1H_ref']) ))


    xl = [np.min(concs.tit)-0.01*tit_rng, np.max(concs.tit)+0.01*tit_rng]
    csp_rng = np.max(mleoutput.csp)-np.min(mleoutput.csp)
    yl_csp = [np.min(mleoutput.csp)-0.05*csp_rng, np.max(mleoutput.csp)+0.05*csp_rng]
    int_rng = np.max(mleoutput.intensity)-np.min(mleoutput.intensity)
    yl_int = [np.min(mleoutput.intensity)-0.05*int_rng, np.max(mleoutput.intensity)+0.05*int_rng]
    
    fname_prefix = config.input_file.split('/')[-1].split('.')[0]

    pdf_name = os.path.join(config.output_dir, fname_prefix + '_MaLICE_fits.pdf')
    with PdfPages(pdf_name) as pdf:
        for residue in residues:
            fig, ax = plt.subplots(ncols=2,figsize=(7.5,2.5))
            ax[0].scatter('tit','csp',data=mleoutput[mleoutput.residue == residue],color='black',s=10)
            ax[0].errorbar('tit','csp',data=mleoutput[mleoutput.residue == residue],yerr=model3b.x[4],color='black',fmt='none',s=16)
            ax[0].plot('tit','cshat',data=fit_data[fit_data.residue == residue])
            ax[1].scatter('tit','intensity',data=mleoutput[mleoutput.residue == residue],color='black',s=10)
            ax[1].errorbar('tit','intensity',data=mleoutput[mleoutput.residue == residue],yerr=model3b.x[3],color='black',fmt='none',s=16)
            ax[1].plot('tit','ihat',data=fit_data[fit_data.residue == residue])
            ax[0].set(xlim=xl, ylim=yl_csp, xlabel='Titrant (μM)', ylabel='CSP (Hz)', title='Residue '+str(residue)+' CSP')
            ax[1].set(xlim=xl, ylim=yl_int, xlabel='Titrant (μM)', ylabel='Intensity', title='Residue '+str(residue)+' Intensity')
            fig.tight_layout()
            pdf.savefig()
            plt.close()


    ## Perform likelihood ratio tests
    print('\n---  Round 4: likelihood ratio test of parameters  ---\n')
    executor = concurrent.futures.ProcessPoolExecutor(config.thread_count)
    futures = [executor.submit(null_calculator, optimizer.mle, config, mleinput, model3b.x, dfs, gvs, bds1, r) for r in residues]
    concurrent.futures.wait(futures)

    dfs['altLL'] = -1 * model3b.fun
    dfs['nullLL'] = [-1*x.result() for x in futures]
    dfs['deltaLL'] = dfs.altLL - dfs.nullLL
    dfs['rank'] = dfs['deltaLL'].rank(ascending=False,na_option='top')
    dfs['p-value'] = (1-stats.chi2.cdf(2*dfs.deltaLL,df=1))*len(dfs)/dfs['rank']
    dfs = dfs.sort_values('deltaLL',ascending=False)
    dfs['sig'] = dfs['p-value'] < 0.01


    ## Compute errors by bootstrapping
    print('\n---  Round 5: bootstrapping to estimate parameter variance  ---\n')

    executor = concurrent.futures.ProcessPoolExecutor(config.thread_count)
    futures = [executor.submit(bootstrapper, optimizer.mle, config, mleinput, model3b.x, gvs, bds1) for i in range(bootstraps)]
    concurrent.futures.wait(futures)
    
    global_params = ['Kd_exp','koff_exp','dR2','Amp','nh_scale','I_noise','CS_noise']
    for k in range(gvs):
        bs_results = [y.result().x[k] for y in futures]
        print(global_params[k]+' = '+str(round(model3b.x[k],2))+' +/- '+str(round(np.std(bs_results),2)))
    
    dfs['stderr'] = [np.std([x.result().x[gvs+r] for x in futures]) for r in range(len(residues))]
    
    
    png_name = os.path.join(config.output_dir, fname_prefix + '_MaLICE_plot.png')
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter('residue','dw',data=dfs[dfs.sig == False],color='black',s=80)
    ax.errorbar('residue','dw',yerr='stderr',data=dfs[dfs.sig == False],color='black',fmt='none',s=20)
    ax.scatter('residue','dw',data=dfs[dfs.sig == True],color='red',s=80)
    ax.errorbar('residue','dw',yerr='stderr',data=dfs[dfs.sig == True],color='red',fmt='none',s=20)
    ax.set(xlim=(np.min(dfs.residue)+1,np.max(dfs.residue)+1),xlabel='Residue',ylabel='Δω (Hz)')
    fig.savefig(png_name,dpi=600,bbox_inches='tight',pad_inches=0)

    ## Print out data
    csv_name = os.path.join(config.output_dir, fname_prefix + '_MaLICE_fits.csv')
    txt_name = os.path.join(config.output_dir, fname_prefix+'_MaLICE_deltaw.txt')
    dfs.to_csv(csv_name,index=False)
    dfs[['residue','dw']].to_csv(txt_name, index=False,header=False)

    endtime = time.time()
    runtime = endtime-starttime
    print('\n\nRun time = '+str(datetime.timedelta(seconds=runtime)).split('.')[0])
if __name__ == "__main__":
    main()
