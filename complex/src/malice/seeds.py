import random

import numpy as np
import pygmo as pg
import tensorflow as tf

_base_seed = None

def set_base_seed(seed):
    if seed is None:
        # Set seed randomly if not explicitly seeded
        seed = random.randint(0, 999999)
    
    _base_seed = seed
    np.random.seed(seed)
    pg.set_global_rng_seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def get_base_seed():
    return _base_seed