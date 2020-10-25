import numpy as np
import scipy.stats as stats

from malice.optimizer import MaliceOptimizer


def walk(optimizer, steps, confidence, min_global, max_global, iterator, steps_to_check=100, abort_threshold=100):

    # Set seed
    np.random.seed(7447+iterator)

    walker = MaliceOptimizer(larmor=optimizer.larmor,
                             gvs=optimizer.gvs,
                             data=optimizer.data,
                             mode='ml_optimization',
                             nh_scale=optimizer.nh_scale,
                             l1_model=optimizer.l1_model,
                             reference=optimizer.reference,
                             ml_model=optimizer.ml_model,
                             lam=0.0,
                             cs_dist='rayleigh')

    min_mcmc = [walker.ml_model[0]-1, walker.ml_model[1]-1, walker.ml_model[2]-20, walker.ml_model[3]/4,
                walker.ml_model[4]/4, walker.ml_model[5]/4] + [0]*(len(walker.ml_model)-walker.gvs)
    max_mcmc = [walker.ml_model[0]+1, walker.ml_model[1]+1, walker.ml_model[2]+20, walker.ml_model[3]*4,
                walker.ml_model[4]*4, walker.ml_model[5]*4] + [max(walker.ml_model[walker.gvs:])*5]*(len(walker.ml_model)-walker.gvs)

    # Fix any of the global bounds that go off into stupid places
    for i in range(walker.gvs):
        if min_mcmc[i] < min_global[i]:
            min_mcmc[i] = min_global[i]
        if max_mcmc[i] > max_global[i]:
            max_mcmc[i] = max_global[i]

    model = list(walker.ml_model)
    model_score = float(walker.fitness(walker.ml_model)[0])
    
    #tolerated_negLL = walker.fitness(walker.ml_model)[0] + stats.chi2.ppf(confidence, df=len(model))/2
    #tolerated_negLL = walker.fitness(walker.ml_model)[0] + 200
    

    modulator = 1.0
    target_accept_rate = 0.7

    
    accepted_steps = []
    temp_failed = 0
    consecutive_failed = 0
    target_ratio = 0
    for i in range(steps):
        prev_model = list(model)

        # For global variables, allow it to walk randomly -/+ 0.4% and dw to move randomly -/+ 0.1 Hz
        ## 200728 I think I need to make the globals move in a bit smaller steps, try reducing to 0.2%
        ## 200731 test1 = prev settings; test2 = up the delta_w step to 0.5 Hz; test3 = 1 Hz; test3s2 = 2Hz
        perturbation = list((np.random.random(walker.gvs)-0.5)/250*np.array(model[:walker.gvs])) + list((np.random.random(len(model[walker.gvs:]))-0.5)*2)
        
        ## I think the next thing I need to try is making the perturbation more fine grain; it's very rough right now
        ## So I think this will be treating the global parameters separately as well as using some kind of scalar basis for delta_w

        ## Initially 0.2% shift for globals, 1% shift for delta_w with an additional +1 variable to ensure that the low values explore some
        ## 200801 Test 1
        '''
        global_scalers = [0.0, 0.0, 0.0, 0.002, 0.002, 0.0]
        deltaw_scalers = [0.000]*len(model[walker.gvs:])
        scaled_perturbation = np.array( global_scalers + deltaw_scalers )
        
        global_offsets = [0.005, 0.005, 0.05, 0, 0, 0.01]
        deltaw_offsets = [1]*(len(model)-walker.gvs)
        offset_perturbation = np.array ( global_offsets + deltaw_offsets )
        '''
        ## 200801 Test 2
        '''
        global_scalers = [0.0, 0.0, 0.0, 0.003, 0.001, 0.0]
        deltaw_scalers = [0.000]*len(model[walker.gvs:])
        scaled_perturbation = np.array( global_scalers + deltaw_scalers )
        
        global_offsets = [0.002, 0.05, 0.1, 0, 0, 0.02]
        deltaw_offsets = [1]*(len(model)-walker.gvs)
        offset_perturbation = np.array ( global_offsets + deltaw_offsets )
        '''
        ## 200801 Test 3
        '''
        global_scalers = [0.0, 0.0, 0.0, 0.004, 0.004, 0.0]
        deltaw_scalers = [0.005]*len(model[walker.gvs:])
        scaled_perturbation = np.array( global_scalers + deltaw_scalers )
        
        global_offsets = [0.003, 0.10, 0.1, 0, 0, 0.02]
        deltaw_offsets = [1]*(len(model)-walker.gvs)
        offset_perturbation = np.array ( global_offsets + deltaw_offsets )
        '''
        ## 200801 Test 4
        '''
        global_scalers = [0.0, 0.0, 0.0, 0.003, 0.003, 0.0]
        deltaw_scalers = [0.003]*len(model[walker.gvs:])
        scaled_perturbation = np.array( global_scalers + deltaw_scalers )
        
        global_offsets = [0.003, 0.12, 0.05, 0, 0, 0.02]
        deltaw_offsets = [1]*(len(model)-walker.gvs)
        offset_perturbation = np.array ( global_offsets + deltaw_offsets )
        '''
        ## 200804 updated after fixing the problem of both gaussian cs errors and the lambda...
        ## Let's just reduce everything 100x and see how that works...
        ## Going back up to only 10X reduction
        ## 200805 - test1 = 10X reduction and doesn't look good. going back to no dilution
        ## 200820 Need to do calibration test, so going back to super duper initial low perturbation, with many steps, and variable
        ## thresholds so that it can more reasonably explore the landscape
        global_scalers = [0.0, 0.0, 0.0, 0.0003, 0.0003, 0.0]
        deltaw_scalers = [0.0003]*len(model[walker.gvs:])
        scaled_perturbation = np.array( global_scalers + deltaw_scalers )
        
        global_offsets = [0.0003, 0.012, 0.005, 0, 0, 0.002]
        deltaw_offsets = [0.1]*(len(model)-walker.gvs)
        offset_perturbation = np.array ( global_offsets + deltaw_offsets )

        perturbation = (2*(np.random.random(len(model))-0.5)) * (scaled_perturbation*model + offset_perturbation)

        model = list(np.array(model) + modulator*perturbation)

        # Check bounds
        for j in range(len(model)):
            if model[j] < min_mcmc[j]:
                model[j] = min_mcmc[j]
            if model[j] > max_mcmc[j]:
                model[j] = max_mcmc[j]
        new_score = float(walker.fitness(model)[0])
        #print(new_score)

        ## 200820 I want to explore from 1 to 100 logL units, so let's adjust the tolerated_negLL to ramp up over the course
        tolerated_negLL = model_score + int( float(i) / (steps/100) ) + 1  
        ## So in a 100K step run, it should spend 1k steps at every threshold
        #tolerated_negLL = 200
        if new_score <= tolerated_negLL:
            #print('passed!')
            #if walker.fitness(model)[0] > 1e4: print('MCMC step >10K')
            accepted_steps.append( (new_score, model) )
            consecutive_failed = 0
        else:
            #print('failed!')
            model = list(prev_model)
            temp_failed += 1
            consecutive_failed += 1

        if i%steps_to_check == 0:
            # Check the current acceptance rate and adjust the modulator to control the step size
            accept_rate = (float(steps_to_check) - temp_failed)/steps_to_check
            if accept_rate < target_accept_rate:
                # Failing too much, need to decrease the step size, reduce by 5%
                modulator = 0.95 * modulator
            else:
                # Accepting at >= target, so increase step size by 5%
                modulator = 1.05 * modulator
            temp_failed = 0  # Reset the counter

        ## 200820 For the moment, let's disable the abortion    
        #if consecutive_failed > abort_threshold:
        #    break

    print('Walk #'+str(iterator+1)+': Accepted '+str(len(accepted_steps))+' of '+str(i+1)+' steps')

    return accepted_steps
