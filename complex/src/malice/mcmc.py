import numpy as np
import scipy.stats as stats

from malice.optimizer import MaliceOptimizer


def walk(optimizer, steps, confidence, min_global, max_global,
         iterator=1, abort_threshold=100):

    walker = MaliceOptimizer(larmor=optimizer.larmor,
                             gvs=optimizer.gvs,
                             data=optimizer.data,
                             mode='ml_optimization',
                             nh_scale=optimizer.nh_scale,
                             l1_model=optimizer.l1_model,
                             reference=optimizer.reference,
                             ml_model=optimizer.ml_model)

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

    tolerated_negLL = walker.fitness(walker.ml_model)[0] + stats.chi2.ppf(confidence, df=1)/2

    model = list(walker.ml_model)
    accepted_steps = []
    consecutive_failed = 0
    for i in range(steps):
        prev_model = list(model)

        # For global variables, allow it to walk randomly -/+ 0.4% and dw to move randomly -/+ 0.1 Hz
        perturbation = list((np.random.random(walker.gvs)-0.5)/125*np.array(model[:walker.gvs])) + list((np.random.random(len(model[walker.gvs:]))-0.5)/5)
        model = list(np.array(model) + perturbation)

        # Check bounds
        for j in range(len(model)):
            if model[j] < min_mcmc[j]:
                model[j] = min_mcmc[j]
            if model[j] > max_mcmc[j]:
                model[j] = max_mcmc[j]

        if walker.fitness(model)[0] <= tolerated_negLL:
            accepted_steps.append(list(model))
            consecutive_failed = 0
        else:
            model = list(prev_model)
            consecutive_failed += 1

        if consecutive_failed > abort_threshold:
            break

    print('Walk #'+str(iterator+1)+': Accepted '+str(len(accepted_steps))+' of '+str(i+1)+' steps')

    return accepted_steps
