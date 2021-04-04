
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



class kinetics_fit(Layer):
    def build(self, input_shape):
        self.Kd_exp = self.add_weight(name='Kd_exp', shape=(1,),
                                      #initializer=Constant(K.log(Kd)/K.log(10.0)),
                                      initializer=RandomUniform(minval=-1, maxval=4),
                                      constraint=MinMaxNorm(min_value=-1, max_value=4, rate=0.1), 
                                      trainable=True)
        self.koff_exp = self.add_weight(name='koff_exp', shape=(1,),
                                        #initializer=Constant(K.log(koff)/K.log(10.0))
                                        initializer=RandomUniform(minval=0, maxval=5),
                                        constraint=MinMaxNorm(min_value=0, max_value=5, rate=0.1), 
                                        trainable=True)

    def call(self, inputs):
        visible, titrant = inputs
        Kd = tf.pow(10.0, self.Kd_exp)
        koff = tf.pow(10.0, self.koff_exp)
        kon = koff/Kd
        
        dimer = ((visible + titrant+Kd) - tf.math.sqrt((visible + titrant + Kd)**2 - 4*visible*titrant))/2
        pb = dimer / visible
        
        ## Assume 1:1 stoich
        free_titrant = titrant - dimer
        kr = koff
        kf = free_titrant * kon
        kex = kr + kf
        
        return pb, kex


class CompLEx_fit(Layer):
    def __init__(self, init_I, larmor):
        super(CompLEx_fit, self).__init__()
        self.init_I = init_I
        self.n_res = len( self.init_I )
        self.larmor = larmor

    def build(self,input_shape):
        self.ref_I = self.add_weight(name='ref_I', shape=(self.n_res,),
                                     initializer=Constant(self.init_I*1.001),
                                     #constraint=MinMaxNorm(min_value=self.init_I/10, max_value=self.init_I*10),
                                     trainable=True )
        init_I_mean = tf.math.reduce_mean( self.init_I )
        init_I_std =  tf.math.reduce_std( self.init_I )
        self.dR2 = self.add_weight(shape=(1,),
                                   #initializer=Constant(dR2),
                                   initializer=RandomUniform(minval=0, maxval=200),
                                   #constraint=MinMaxNorm(min_value=0, max_value=200), 
                                   trainable=True )
        self.amp_scaler = self.add_weight(name='amp_scaler', shape=(1,),
                                          #initializer=Constant(amp_scaler),#
                                          initializer=RandomNormal(mean=float(5*init_I_mean), stddev=float(5*init_I_std)), 
                                          constraint=MinMaxNorm(min_value=init_I_mean/10.0, max_value=init_I_mean*10.0), 
                                          trainable=True )
        
        self.delta_w = self.add_weight(name='delta_w', shape=(self.n_res,),
                                       initializer=Constant(self.larmor/100),
                                       constraint=MinMaxNorm(min_value=0, max_value=6.0*self.larmor, rate=0.1),
                                       trainable=True)
    def call(self, inputs): #inputs):
        resn_array, visible, pb, kex = inputs
        pa = 1 - pb
        
        dw = K.sum( self.delta_w * resn_array, axis=1, keepdims=True )
        I = K.sum( self.ref_I * resn_array, axis=1, keepdims=True )
        amp = self.amp_scaler / tf.math.reduce_mean(visible)
        
        broad_denom = (kex**2 + (1-5*pa*pb)*(dw**2))**2 + 4*pa*pb*(1-4*pa*pb)*(dw**4)
        
        ## Abergel-Palmer approximations
        i_broad = pa*pb*(dw**2)*kex * (kex**2 + (1-5*pa*pb)*(dw**2)) / broad_denom
        ihat_ap = I / ( pa + pb + I*(pb*self.dR2 + i_broad)/amp )
        cs_broad = pa*pb*(pa-pb)*(dw**3) * (kex**2 + (1-3*pa*pb)*(dw**2)) / broad_denom
        cshat_ap = pb*dw - cs_broad
        
        ## Slow exchange
        #cshat_slow = 0.0
        #ihat_slow = pa * I
        #ap_select = K.cast(dw/kex < 2,'float32'), #'float32')
        #non_ap_select = K.cast(dw/kex >= 2,'float32')
        #ihat = ap_select*ihat_ap + non_ap_select*ihat_slow
        #cshat = ap_select*cshat_ap + non_ap_select*cshat_slow
        #print(ap_select.shape)
        return ihat_ap, cshat_ap
        


class CSP_fit(Layer):
    def __init__(self, init_N, init_H, larmor):
        super(CSP_fit, self).__init__()
        self.n_res = len( init_N )
        self.init_N = init_N
        self.init_H = init_H
        self.larmor = larmor
    
    def build(self, input_shape):
        self.ref_N = self.add_weight(shape=(self.n_res,),
                                     initializer=Constant(self.init_N+0.005),
                                     #constraint=MinMaxNorm(min_value=self.init_N-0.2, max_value=self.init_N+0.2), 
                                     trainable=False)
        self.ref_H = self.add_weight(shape=(self.n_res,),
                                     initializer=Constant(self.init_H+0.001),
                                     #constraint=MinMaxNorm(min_value=self.init_H-0.05, max_value=self.init_H+0.05),
                                     trainable=False)
                                     
    def call(self, inputs):
        resn_array, N_obs, H_obs = inputs
        N_ref = K.sum( self.ref_N * resn_array, axis=1, keepdims=True)
        H_ref = K.sum( self.ref_H * resn_array, axis=1, keepdims=True)
        csp = K.sqrt((0.2*(N_obs-N_ref))**2 + (H_obs-H_ref)**2)*self.larmor
        return csp
    
    
class int_negLogL(Layer):
    def __init__(self, init_I):
        super(int_negLogL, self).__init__()
        self.init_I = init_I
    
    def build(self,input_shape):
        init_I_mean = float(tf.math.reduce_mean( self.init_I ))
        self.I_noise = self.add_weight( shape=(1,),
                                        initializer=Constant(init_I_mean/20.0),#RandomUniform(minval=init_I_mean/50, maxval=init_I_mean/4),  
                                        constraint=MinMaxNorm(min_value=init_I_mean/10.0, max_value=init_I_mean*10.0), 
                                        trainable=False )
        
    def call(self, inputs):
        y_true, y_pred = inputs
        dist = tfp.distributions.Normal(loc=y_pred, scale=self.I_noise)
        return -1*dist.log_prob(y_true)

class cs_negLogL(Layer):
    def __init__(self, larmor):
        super(cs_negLogL, self).__init__()
        self.larmor = larmor
    
    def build(self, input_shape):
        self.cs_noise = self.add_weight( shape=(1,), 
                                         initializer=Constant(self.larmor/250),#RandomUniform(minval=self.larmor/5000, maxval=self.larmor/500), 
                                         constraint=MinMaxNorm(min_value=self.larmor/4500, max_value=self.larmor/50), 
                                         trainable=False )
    
    def call(self, inputs):
        y_true, y_pred = inputs
        dist = tfp.distributions.Chi2(df=2)
        z = K.abs(y_true - y_pred)/self.cs_noise
        return -1*(dist.log_prob(z) - tf.math.log(self.cs_noise))

def identity_loss(y_true, loss):
    return loss

def sum_loss(y_true,y_pred):
    return K.sum(y_pred)