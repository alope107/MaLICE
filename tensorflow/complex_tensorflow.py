## CompLEx Tensorflow implementation


## Libraries
import sys, argparse
import numpy as np, pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from layers import *

import keras
from keras.layers import Input, Layer, Add
from keras.models import Model
from keras.initializers import VarianceScaling, RandomUniform, RandomNormal, Constant
from keras.constraints import MinMaxNorm
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Adamax
import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K

from malice.seeds import set_base_seed



def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', 
                        type=str, 
                        help='path to the CSV to import')
    parser.add_argument('--larmor',
                        type=float,
                        help="Larmor frequency (MHz) of 1H in the given magnetic field.",
                        default=500)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to store output files. Creates if non-existent.',
                        default="output")
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
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    #make_output_dir(args.output_dir)
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
    data = data.sort_values(['residue','titrant'])
    residues = sorted(list(data.residue.unique()))
    reference_points = pd.DataFrame()
    for res in residues:
        resdata = data.copy()[data.residue == res]
        # Use the lowest titration point (hopefully zero) for the reference
        min_titrant_conc = resdata.titrant.min()
        reference_points = reference_points.append(resdata.loc[resdata.titrant == min_titrant_conc, ['residue', '15N', '1H', 'intensity']].mean(axis=0), ignore_index=True)

    return data, reference_points, residues


def run_malice(config):

    # Important variables
    larmor = config.larmor
    gvs = 6
    lam = 0.0003
    nh_scale = 0.2  # Consider reducing to ~0.14

    user_data, initial_reference, residues = parse_input(config.input_file)

    ## Parse data into tensors
    tensors = {}
    tensors['residue'] = np.asarray( [ [1 if residues[i] == r else 0 for i in range(len(residues))] 
                                        for r in user_data.residue ], 'float32' )
    for dtype in ['15N','1H','intensity','visible','titrant']:
        tensors[dtype] = np.reshape( np.asarray( list(user_data[dtype]), 'float32' ), (len(user_data),1) )

    initials = {}
    for dtype in ['15N','1H','intensity']:
        initials[dtype] = np.asarray( list(initial_reference[dtype]), 'float32' )

    #fname_prefix = config.input_file.split('/')[-1].split('.')[0]


    ## Model
    visible_input = Input(shape=(1,), name='visible')
    titrant_input = Input(shape=(1,), name='titrant')
    N15_input = Input(shape=(1,), name='15N')
    H1_input = Input(shape=(1,), name='1H')
    Int_input = Input(shape=(1,), name='intensity')
    residue_input = Input(shape=(104,), name='residue_array')

    pb, kex = kinetics_fit()([visible_input, titrant_input])
    ihat,cshat = CompLEx_fit(initials['intensity'], larmor)([residue_input,visible_input,pb,kex])

    csobs = CSP_fit(initials['15N'],initials['1H'], larmor)([residue_input,N15_input,H1_input])

    int_loss = int_negLogL(initials['intensity'])([Int_input,ihat])
    cs_loss = cs_negLogL(larmor)([csobs,cshat])

    summed_loss = Add()([int_loss,cs_loss])

    model = Model( inputs=[residue_input, N15_input, H1_input, Int_input, visible_input, titrant_input],
                outputs=summed_loss)
    model.compile(optimizer=Adam(learning_rate=4e-2), loss=sum_loss)

    decay = ReduceLROnPlateau(monitor='loss', patience=50, factor=0.5)
    history = model.fit( [tensors[x] for x in ['residue','15N','1H','intensity','visible','titrant']],
                        np.zeros(len(tensors['15N'])),
                        epochs=100, batch_size=24, verbose=1, shuffle=True,
                        callbacks=[decay] )


    ## Report some quick diagnostics
    Kd_exp, koff_exp = [float(x) for x in model.layers[3].weights]
    dR2, amp_scaler = [float(x) for x in model.layers[7].weights[1:3]]
    deltaw_array = np.array(model.layers[7].weights[3])
    print('Kd = '+format(np.power(10.0,Kd_exp),'.1f'))
    print('koff = '+format(np.power(10.0,koff_exp),'.1f'))
    print('dR2 = '+format(dR2,'.2f'))
    print(deltaw_array)

    ## bleh
    return 0


if __name__ == "__main__":
    main()


