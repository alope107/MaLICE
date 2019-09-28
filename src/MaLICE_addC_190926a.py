### MaLICE v190923-0903
### First standalone script version

## Notable features:	- Split the global variable and delta_w optimization
##						  from reference peak optimization
##						- Currently using additive C
##							- Need to go back validate this is best

## Import libraries
import sys, itertools, time, datetime
import numpy as np, pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize,basinhopping,differential_evolution
from matplotlib import pyplot as plt
import matplotlib.cm as cm

starttime = time.time()

## Read in data
#mleinput = pd.read_csv(sys.argv[1])

## Currently going to be lazy and just have VERL and FITZAP data pre-loaded
## but eventually want to have the code be brought in
cwd = '../data/'
verl = pd.read_csv(cwd+'verl_190726.csv')

fitzap = pd.read_csv(cwd+'fitzap_190821.csv')
fitzap = fitzap.copy()[[x[-3:] == 'N-H' for x in fitzap.residue]]
fitzap['residue'] = [int(x[1:-3]) for x in fitzap.residue]

#mleinput = verl.copy()  # Start with VERL data
mleinput = fitzap.copy()
'''
mleinput = pd.read_csv('abd2_fMaLICE_190924.csv')
mleinput = mleinput[mleinput.residue.between(289,356,inclusive=True)]
mleinput = mleinput[mleinput.intensity > 0.3]
mleinput = mleinput[mleinput.tit < 500]
'''
residues = list(mleinput.groupby('residue').groups.keys())
resgrouped = mleinput.loc[mleinput.tit == 0,['residue','15N','1H','intensity']]
i_noise_est = np.mean(mleinput.intensity)/10


## Important variables
larmor = 500
nh_scale = 0.2
gvs = 6
lam = 0.01
threads = 10

pop_iter = 10

## Stage 1 - initial global and delta_w optimization
## Equation 1
def mle_lambda(params):
	Kd_exp, koff_exp, C, i_noise, cs_noise, amp = params[0], params[1], params[2], params[3], params[4], params[5]
	
	Kd = np.power(10,Kd_exp)
	koff = np.power(10,koff_exp)
	kon = koff/Kd
	
	resparams = resgrouped.copy().rename(columns={'intensity':'I_ref','15N':'15N_ref','1H':'1H_ref'})
	resparams['dw'] = params[gvs:]
	
	df = pd.merge(mleinput,resparams,on='residue')
	
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
	ihat = df.I_ref/( pa + pb + df.I_ref*(pb*C + i_broad)/amp )
	logLL_int = np.sum( stats.norm.logpdf(df.intensity, loc=ihat, scale=i_noise) )
	
	#Calculate cs likelihood
	cs_broad = pa*pb*(pa-pb)*np.power(df.dw,3) * (np.square(kex)+(1-3*pa*pb)*np.square(df.dw))/broad_denom
	cshat = pb*df.dw - cs_broad
	csobs = larmor*(np.sqrt( nh_scale*np.square(df['15N'] - df['15N_ref']) + np.square(df['1H'] - df['1H_ref']) ))
	logLL_cs = np.sum( stats.norm.logpdf(csobs, loc=cshat, scale=cs_noise) )
	
	negLL = -1*(logLL_int + logLL_cs - lam*np.sum(np.abs(df.dw)))
	
	return(negLL)

## Counter 1
def counter_1(xk,convergence=1e-7):
	global i
	if i%1000 == 0:
		print(str(i).ljust(8)+'Score: '+str(round(mle_lambda(xk),2)).ljust(12)+
			  '-logL: '+str(round(mle_lambda(xk)-lam*np.sum(xk[gvs:]),2)).ljust(12)+
			  'Kd: '+str(round(np.power(10,xk[0]),1)).ljust(10)+
			  'C: '+str(round(xk[2],2)).ljust(8)+
			  'max_dw: '+str(round(np.max(xk[int(6+3*(len(xk)-6)/4):]),2)))
	i+=1

## Perform 10 replicates of 20 member populations
print('\n---  Round 1: initial global variable and delta w optimization  ---\n')
models1 = []
for iteration in range(pop_iter):
	## Let's define a new starting population every time to try to introduce more coverage of the space
	pop = []
	pop_size = 20  #small starting pop
	for x in range(pop_size):
		Kd_exp_random = list(np.random.random(1)*7-3)   # Will span from 1 nM to 10 mM
		kex_exp_random = list(np.random.random(1)*2+3)  # Will span from 1 kHz to 100 kHz
		C_random = list(np.random.random(1)*200)			# 0 - 200 Hz
		i_noise_random = list(np.mean(mleinput.intensity)/(np.random.random(1)*46+4)) # 1/4 to 1/50th of mean intensity
		cs_noise_random = list(larmor/(np.random.random(1)*4450+50)) # larmor / 50-4500 -- rough range of digital res
		amp_random = [np.random.normal(np.mean(mleinput.intensity),np.std(mleinput.intensity)) * 20]
			# random amp logic is that since amp = intensity * lw, lets just randomly sample something from the reasonable intensity
			# range and multiply by 20, which is probably a decent enough guess of typical protein linewidths

		dw_random = list( 0.1*larmor * np.random.random(len(residues)) ) ## Every delta_w is 0-0.1 ppm CSP

		pop.append(Kd_exp_random + kex_exp_random + C_random + i_noise_random + cs_noise_random + amp_random + dw_random)


	mininit = [-3, 1, 0, i_noise_est/10, larmor/4500, np.min(mleinput.intensity)/10] + list([0]*len(residues))
	maxinit = [4, 8, 200, i_noise_est*10, larmor/50, np.max(mleinput.intensity)*200] + list([6*larmor]*len(residues))

	bds1 = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])
	
	## Run the job
	print('Round 1 - Population # '+str(iteration+1))
	i = 0
	initfit = differential_evolution(mle_lambda, bds1, init = pop, updating='deferred', 
									 workers = threads, mutation=(0.5,1.9),maxiter = 100000, 
									 strategy = 'best1bin', polish = False, recombination = 0.7, 
									 tol=1e-6, disp=False, callback=counter_1)
	print('\n'+str(round(initfit.fun - lam*np.sum(initfit.x[gvs:]),2))+'\n')
	models1.append(initfit)

## Sort the results by -logL and use the best one for reference peak optimization
models1.sort(key=lambda y:y.fun - lam*np.sum(y.x[gvs:]))
model1 = models1[0]

## Stage 2 - reference peak optimization
print('\n---  Round 2: reference peak optimization  ---\n')

## Equation 2
def refpeak_opt(params):
	Kd_exp = model1.x[0]
	koff_exp = model1.x[1]
	C = model1.x[2]
	i_noise = model1.x[3]
	cs_noise = model1.x[4]
	amp = model1.x[5]
	
	Kd = np.power(10,Kd_exp)
	koff = np.power(10,koff_exp)
	kon = koff/Kd
	
	resparams = pd.DataFrame({'residue':residues,'15N_ref':params[:int(len(params)/3)],
							  '1H_ref':params[int(len(params)/3):2*int(len(params)/3)],
							  'I_ref':params[2*int(len(params)/3):],'dw':model1.x[gvs:]})
	
	df = pd.merge(mleinput,resparams,on='residue')
	
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
	ihat = df.I_ref/( pa + pb + df.I_ref*(pb*C + i_broad)/amp )
	logLL_int = np.sum( stats.norm.logpdf(df.intensity, loc=ihat, scale=i_noise) )
	
	#Calculate cs likelihood
	cs_broad = pa*pb*(pa-pb)*np.power(df.dw,3) * (np.square(kex)+(1-3*pa*pb)*np.square(df.dw))/broad_denom
	cshat = pb*df.dw - cs_broad
	csobs = larmor*(np.sqrt( nh_scale*np.square(df['15N'] - df['15N_ref']) + np.square(df['1H'] - df['1H_ref']) ))
	logLL_cs = np.sum( stats.norm.logpdf(csobs, loc=cshat, scale=cs_noise) )
	
	negLL = -1*(logLL_int + logLL_cs)
	
	return(negLL)

## Define the counter
def counter_2(xk,convergence=1e-7):
	global i
	if i%1000 == 0:
		print(str(i).ljust(8)+'-logL: '+str(round(refpeak_opt(xk),2)).ljust(12))
	i+=1


## Set up replicate populations and run
models2 = []
for z in range(pop_iter):
	pop = []
	pop_size = 20  #small starting pop 
	for x in range(pop_size):
		N_random = list( np.array(resgrouped['15N']) + np.random.normal(0,model1.x[4]/nh_scale,len(residues)) )
		H_random = list( np.array(resgrouped['1H']) + np.random.normal(0,model1.x[4],len(residues)) )  
		Iref_random = list( np.array(resgrouped['intensity']) + np.random.normal(0,model1.x[3],len(residues)) )

		pop.append(N_random + H_random + Iref_random)
	pop.append( list(resgrouped['15N']) + list(resgrouped['1H']) + list(resgrouped['intensity']) )

	mininit =  list(np.array(resgrouped['15N']) - 0.2) + list(
					np.array(resgrouped['1H']) - 0.05) + list(np.array(resgrouped.intensity)/10) 
	maxinit =  list(np.array(resgrouped['15N']) + 0.2) + list(
					np.array(resgrouped['1H']) + 0.05) + list(np.array(resgrouped.intensity)*10) 

	bds2 = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])
	
	print('Round 2 - Population #'+str(z+1))
	i = 0
	ref_opt = differential_evolution(refpeak_opt, bds2, init = pop, updating='deferred', 
									 workers = threads, mutation=(0.5,1.9), maxiter = 100000, 
									 strategy = 'best1bin', polish = False, recombination = 0.7,
									 tol=1e-7, disp=False, callback=counter_2)
	print('\n'+str(round(ref_opt.fun,2))+'\n')
	models2.append(ref_opt)

## Sort the results by -logL and use the best one for final polishing
models2.sort(key=lambda y:y.fun)
model2 = models2[0]

## Polish the reference peaks
model2opt = minimize(refpeak_opt, model2.x, method='SLSQP',bounds=bds2,
					 tol=1e-7, options={'disp':True,'maxiter':10000})

## Stage 3 - polish off the model with gradient minimization
print('\n---  Round 3: gradient minimization of global variables and delta w  ---\n')


## Equations 3a and 3b -- first to optimize by scale, then fine tuning
def mle_reduced(params):
	Kd_exp, koff_exp, C, i_noise, cs_noise, amp, scale = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
	
	Kd = np.power(10,Kd_exp)
	koff = np.power(10,koff_exp)
	kon = koff/Kd
	
	resparams = pd.DataFrame({'15N_ref':model2opt.x[:int(len(model2opt.x)/3)], 
							  '1H_ref':model2opt.x[int(len(model2opt.x)/3):2*int(len(model2opt.x)/3)],
							  'I_ref':model2opt.x[2*int(len(model2opt.x)/3):],
							  'residue':residues})
	resparams['dw'] = model1.x[gvs:]*scale
	
	df = pd.merge(mleinput,resparams,on='residue')
	
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
	ihat = df.I_ref/( pa + pb + df.I_ref*(pb*C + i_broad)/amp )
	logLL_int = np.sum( stats.norm.logpdf(df.intensity, loc=ihat, scale=i_noise) )
	
	#Calculate cs likelihood
	cs_broad = pa*pb*(pa-pb)*np.power(df.dw,3) * (np.square(kex)+(1-3*pa*pb)*np.square(df.dw))/broad_denom
	cshat = pb*df.dw - cs_broad
	csobs = larmor*(np.sqrt( nh_scale*np.square(df['15N'] - df['15N_ref']) + np.square(df['1H'] - df['1H_ref']) ))
	logLL_cs = np.sum( stats.norm.logpdf(csobs, loc=cshat, scale=cs_noise) )
	
	negLL = -1*(logLL_int + logLL_cs)
	
	return(negLL)

def mle_full(params):
	Kd_exp, koff_exp, C, i_noise, cs_noise, amp = params[0], params[1], params[2], params[3], params[4], params[5]
	
	Kd = np.power(10,Kd_exp)
	koff = np.power(10,koff_exp)
	kon = koff/Kd
	
	resparams = pd.DataFrame({'15N_ref':model2opt.x[:int(len(model2opt.x)/3)], 
							  '1H_ref':model2opt.x[int(len(model2opt.x)/3):2*int(len(model2opt.x)/3)],
							  'I_ref':model2opt.x[2*int(len(model2opt.x)/3):],
							  'residue':residues})
	resparams['dw'] = params[gvs:]
	
	df = pd.merge(mleinput,resparams,on='residue')
	
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
	ihat = df.I_ref/( pa + pb + df.I_ref*(pb*C + i_broad)/amp )
	logLL_int = np.sum( stats.norm.logpdf(df.intensity, loc=ihat, scale=i_noise) )
	
	#Calculate cs likelihood
	cs_broad = pa*pb*(pa-pb)*np.power(df.dw,3) * (np.square(kex)+(1-3*pa*pb)*np.square(df.dw))/broad_denom
	cshat = pb*df.dw - cs_broad
	csobs = larmor*(np.sqrt( nh_scale*np.square(df['15N'] - df['15N_ref']) + np.square(df['1H'] - df['1H_ref']) ))
	logLL_cs = np.sum( stats.norm.logpdf(csobs, loc=cshat, scale=cs_noise) )
	
	negLL = -1*(logLL_int + logLL_cs)
	
	return(negLL)


## Run the 3a scaling optimization
# Bounds
mininit = [model1.x[0]-1, model1.x[1]-1, model1.x[2]-20, model1.x[3]/4, model1.x[4]/4, 
		   model1.x[5]/4, 0.1]
maxinit = [model1.x[0]+1, model1.x[1]+1, model1.x[2]+20, model1.x[3]*4, model1.x[4]*4, 
		   model1.x[5]*4, 10]

# Fix any of the global bounds that go off into stupid places
for i in range(gvs):
	if mininit[i] < bds1[i][0]:	mininit[i] = bds1[i][0]
	if maxinit[i] > bds1[i][1]:	maxinit[i] = bds1[i][1]

bds3a = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])

init3a = list(model1.x[:gvs]) + [1]

model3a = minimize(mle_reduced, init3a, method='SLSQP',bounds=bds3a,
				   tol=1e-7,options={'disp':True,'maxiter':10000})


## Run the 3b fine tuning optimization
# Bounds
mininit = [model3a.x[0]-0.2, model3a.x[1]-0.2, model3a.x[2]-10, model3a.x[3]/1.2,
		   model3a.x[4]/1.2, model3a.x[5]/1.2] + [0]*len(residues)
maxinit = [model3a.x[0]+0.2, model3a.x[1]+0.2, model3a.x[2]+10, model3a.x[3]*1.2,
		   model3a.x[4]*1.2, model3a.x[5]*1.2] + list(np.array(model1.x[gvs:])*model3a.x[-1]*2)

# Fix any of the global bounds that go off into stupid places
for i in range(gvs):
	if mininit[i] < bds1[i][0]:	mininit[i] = bds1[i][0]
	if maxinit[i] > bds1[i][1]:	maxinit[i] = bds1[i][1]

bds3b = tuple([(mininit[x],maxinit[x]) for x in range(len(mininit))])

init3b = list(model3a.x[:gvs]) + list(np.array(model1.x[gvs:])*model3a.x[-1])

# Full minimization
model3b = minimize(mle_full, init3b, method='SLSQP', bounds=bds3b,
				  tol=1e-7, options={'disp':True,'maxiter':10000})

print('\nFinal Score = '+str(round(model3b.fun,2)))


## Let's go ahead and print some results + save a figure
print('\n\tKd = '+str(round(np.power(10,model3b.x[0]),2))+
	  '\n\tKex = '+str(round(np.power(10,model3b.x[1]),2))+
	  '\n\tC = '+str(round(model3b.x[2],2)))

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(residues,model3b.x[gvs:])
#fig.savefig('verl190923.png',dpi=600,bbox_inches='tight',pad_inches=0)


endtime = time.time()
runtime = endtime-starttime
print('Run time = '+str(datetime.timedelta(seconds=runtime)).split('.')[0])




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

def model_fitter(df):
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

fit_data = model_fitter(fitter_input)

mleoutput = pd.merge(mleinput,resparams,on='residue')
mleoutput['csp'] = larmor*(
					np.sqrt( 
						nh_scale*np.square(mleoutput['15N'] - mleoutput['15N_ref']) + 
							     np.square(mleoutput['1H'] - mleoutput['1H_ref']) ))


xl = [np.min(concs.tit)-0.1*tit_rng, np.max(concs.tit)+0.1*tit_rng]

with PdfPages('abd2_fits_no5x.pdf') as pdf:
    for residue in residues:
        fig, ax = plt.subplots(ncols=2,figsize=(7.5,2.5))
        ax[0].scatter('tit','csp',data=mleoutput[mleoutput.residue == residue])
        ax[0].plot('tit','cshat',data=fit_data[fit_data.residue == residue])
        ax[1].scatter('tit','intensity',data=mleoutput[mleoutput.residue == residue])
        ax[1].plot('tit','ihat',data=fit_data[fit_data.residue == residue])
        ax[0].set(xlim=xl,title='Residue '+str(residue))
        ax[1].set(xlim=xl,title='Residue '+str(residue))
        pdf.savefig()
        plt.close()
