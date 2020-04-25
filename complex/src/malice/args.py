import argparse

def parse_args():
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
                        help='random seed (still not deterministic if using multiple threads)',
                        default=None)
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