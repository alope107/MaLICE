## CompLEx Tensorflow implementation
import sys, argparse
import numpy as np, pandas as pd

from layers import *

from keras.layers import Input, Layer, Add
from keras.models import Model
from keras.initializers import RandomUniform, RandomNormal, Constant
from keras.constraints import MinMaxNorm
from keras.callbacks import Callback, TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

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


## DBW 210504
## USED FOR GENERATING BATCHES BASED ON RESIDUES RATHER THAN INDIVIDUAL OBSERVATIONS
## PROBABLY DELETE LATER
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input, batch_size=3, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.input = input
        self.indices = list(range(len(input)))
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, index):
        index = self.index[index*self.batch_size:(index+1)*self.batch_size]
        ## Ok, so index in this case is the residue numbers randomized
        ## Concatenate the variable columns togther for the number of indices selected

        n_columns = len( self.input[ index[0] ] )
        x = [ np.concatenate([self.input[idx][i] for idx in index]) for i in range(n_columns) ]
        y = np.zeros( len(x[0]) )
        return x, y


    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

#def validate_bounds( parameters, bounds_dict ):
class VerifyBounds(Callback):
    def __init__(self, bounds_dict):
        super(VerifyBounds, self).__init__()

        ## DO SOMETHING WITH BOUNDS_DICT
        self.bounds_dict = bounds_dict
    
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        # Weights is a 10 member list of arrays in the following order
        # [ 'Kd_exp', 'koff_exp', 'ref_I_offset', 'dR2', 'amp_scaler', 'delta_w', 'ref_N_offset', 'ref_H_offset', 'I_noise', 'cs_noise' ]
        # Bounds dict has been ordered to match
        for idx, var in enumerate(self.bounds_dict):
            if np.any(weights[idx] < self.bounds_dict[var][0]) or np.any(weights[idx] > self.bounds_dict[var][1]):
                print(var+' is out of bounds')
                self.model.stop_training = True


        print(len(weights))
        ## Should be able to get weights using self.model
        ## If there is a violation of the bounds, can set flag self.model.stop_training = True










def run_malice(config):

    # Important variables
    larmor = config.larmor
    gvs = 6
    lam = 0.0003
    nh_scale = 0.2  # Consider reducing to ~0.14

    user_data, initial_reference, residues = parse_input(config.input_file)

    init_intensity_mean = np.mean( initial_reference.intensity )

    ## Define lower/upper bounds for parameters
    ## Consider making togglable
    bounds = { 'Kd_exp' : (-1, 4),
               'koff_exp' : (0, 5),
               'I_offset' : (-0.1*init_intensity_mean, 0.1*init_intensity_mean),
               'dR2' :    (0.01, 200),
               'amp_scaler' : (init_intensity_mean, init_intensity_mean*100),
               'delta_w' : (0.0, larmor*6.0),
               'N_offset' : (-0.2, 0.2),
               'H_offset' : (-0.05, 0.05),
               'I_noise': (init_intensity_mean/100, init_intensity_mean/5),
               'cs_noise': (larmor/4500, larmor/50) }



    ## Parse data into tensors
    ## DBW 210504 -- group all rows together by residue so that they can be called together
    ## DISABLING FOR NOW
    '''
    tensors_by_residue = []
    for r in residues:
        residue_tensors = []
        residue_data = user_data[user_data.residue == r]
        residue_tensors.append( np.asarray( [[1 if residues[i] == r else 0 
                                            for i in range(len(residues))]]*len(residue_data), 
                                            'float32' ) )
        for dtype in ['15N','1H','intensity','visible','titrant']:
            residue_tensors.append( np.reshape( np.asarray( list(residue_data[dtype]), 'float32' ), 
                                    (len(residue_data),1) ) )
        tensors_by_residue.append( residue_tensors )
    '''

    ## Simplified tensor input
    x_input = []
    x_input.append( np.asarray( [ [1 if residues[i] == r else 0 for i in range(len(residues))] 
                                  for r in user_data.residue ], 'float32' )  )
    for dtype in ['15N','1H','intensity','visible','titrant']:
        x_input.append( np.reshape( np.asarray( list(user_data[dtype]), 'float32' ), (len(user_data),1) ) )                       
    y_output = np.zeros( len(x_input[1]) )
    #tensors = {}
    #tensors['residue'] = np.asarray( [ [1 if residues[i] == r else 0 for i in range(len(residues))] 
    #                                    for r in user_data.residue ], 'float32' )
    #for dtype in ['15N','1H','intensity','visible','titrant']:
    #    tensors[dtype] = np.reshape( np.asarray( list(user_data[dtype]), 'float32' ), (len(user_data),1) )

  

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

    pb, kex = KineticsFit(bounds)([visible_input, titrant_input])
    ihat,cshat = ComplexFit(initials['intensity'], larmor, bounds)([residue_input,visible_input,pb,kex])
    csobs = CspFit(initials['15N'],initials['1H'], larmor, bounds)([residue_input,N15_input,H1_input])

    int_loss = IntNegLogL(initials['intensity'],bounds)([Int_input,ihat])
    cs_loss = CsNegLogL(larmor,bounds)([csobs,cshat])
    summed_loss = Add()([int_loss,cs_loss])

    model = Model( inputs=[residue_input, N15_input, H1_input, Int_input, visible_input, titrant_input],
                outputs=summed_loss)
    model.compile(optimizer=Adam(learning_rate=4e-2), loss=sum_loss)

    decay = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.5)
    verify_bounds = VerifyBounds(bounds)
    terminate_nan = TerminateOnNaN()
    history = model.fit( x_input, y_output,
                         #DataGenerator(tensors_by_residue),
                         #[tensors[x] for x in ['residue','15N','1H','intensity','vigit stsible','titrant']],
                         #np.zeros(len(tensors['15N'])),
                         epochs=100, batch_size=12, verbose=1, shuffle=True,
                         callbacks=[decay, verify_bounds, terminate_nan] )


    ## Report some quick diagnostics
    Kd_exp, koff_exp = [float(x) for x in model.layers[3].weights]
    dR2, amp_scaler = [float(x) for x in model.layers[7].weights[1:3]]
    deltaw_array = np.array(model.layers[7].weights[3])
    print('Kd = '+format(np.power(10.0,Kd_exp),'.1f'))
    print('koff = '+format(np.power(10.0,koff_exp),'.1f'))
    print('dR2 = '+format(dR2,'.2f'))

    print(f'{[Kd_exp, koff_exp, dR2, amp_scaler, list(deltaw_array)]}')

    return Kd_exp, koff_exp, dR2, amp_scaler, list(deltaw_array)


if __name__ == "__main__":
    main()


