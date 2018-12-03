import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.engine.topology import Layer
from keras import regularizers
from keras.regularizers import Regularizer
from keras import initializers
from keras import activations

import numpy as np
import tensorflow as tf
import cv2
import scipy.ndimage.filters as scifilters
from skimage.util import view_as_windows


## General Utility Functions
def channels_to_complex(X):
    return tf.complex(X[..., 0], X[..., 1])

def channels_to_complex_np(X):
    return X[..., 0] + 1j * X[..., 1]

def complex_to_channels(Z):
    RE = tf.real(Z)
    IM = tf.imag(Z)

    if Z.get_shape()[-1] == 1:
        RE = tf.squeeze(RE, [-1])
        IM = tf.squeeze(IM, [-1])

    return tf.stack([RE, IM], axis=-1)

def complex_to_channels_np(Z):
    RE = np.real(Z)
    IM = np.imag(Z)

    if Z.shape[-1] == 1:
        RE = np.squeeze(RE, (-1))
        IM = np.squeeze(IM, (-1))

    return np.stack([RE, IM], axis=-1)

def real_to_channels(X):
    import math 
    
 #   max_val = np.max(X)
    # make phase proportional to amplitude
 #   R = X
 #   THETA = 2*math.pi*X/max_val
 #   X_c = polar_to_rect(np.stack([R, THETA], axis=-1))
                        
    # Create complex with zero imaginary part                        
    X_c = tf.complex(X, 0.0) # if forcing phase to be zero
    return complex_to_channels(X_c)

def real_to_channels_np(X):
    import math
    # Create complex with zero imaginary part
    X_c = X + 0.j
   # max_val = np.max(X)
    # make phase proportional to amplitude
 #   R = X
 #   THETA = 2.0*math.pi*X/max_val
    #X_c = polar_to_rect_np(np.stack([R, THETA], axis=-1))
 #   X_c = R*tf.exp(1j*THETA)
    return complex_to_channels_np(X_c)


def rect_to_polar(X):
    Z = channels_to_complex(X)
    R = tf.abs(Z)
    THETA = tf.angle(Z)

    if Z.shape[-1] == 1:
        R = tf.squeeze(R, (-1))
        THETA = tf.squeeze(THETA, (-1))

    return tf.stack([R, THETA], axis=-1)




def rect_to_polar_np(X):
    Z = channels_to_complex_np(X)
    R = np.abs(Z)
    THETA = np.angle(Z)

    if Z.shape[-1] == 1:
        R = np.squeeze(R, (-1))
        THETA = np.squeeze(THETA, (-1))

    return np.stack([R, THETA], axis=-1)


def polar_to_rect(X):
    r, theta = X[..., 0], X[..., 1]
    return tf.stack([r*tf.cos(theta), r*tf.sin(theta)], axis=-1)

def polar_to_rect_np(X):
    return complex_to_channels_np(X[..., 0] * np.exp(1j * X[..., 1]))

def real_to_channels_prop_np(r, max_val, max_phase_delay):
    theta = -max_phase_delay*np.pi*r/max_val
    polar = np.stack([r, theta], axis=-1)
    rect = polar_to_rect_np(polar)
    return rect

def real_to_channels_prop(r, max_val, max_phase_delay):
    theta = -max_phase_delay*np.pi*r**2/max_val
    polar = tf.stack([r.astype('float32'), theta.astype('float32')], axis=-1)
    rect = polar_to_rect(polar)
    return rect



## Sparsity functions
def num_less_than_eps(x, eps):
    return (x < eps).sum()

def num_abs_less_than_eps(x, eps):
    x_c = x[..., 0] + 1j*x[..., 1]
    return num_less_than_eps(abs(x_c), eps)

def zero_elems_less_than_eps(x, eps):
    arr = np.copy(x) # This avoids actually changing the argument x
    arr[np.where(arr < eps)] = 0
    return arr


## Regularizers
class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, W):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AmplitudeRegL1(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        return self.lamb * K.sum(tf.abs(complex_W))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class AmplitudeRegL2(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        return self.lamb * K.sum(K.square(tf.abs(complex_W)))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class PhaseReg(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        return K.sum(self.lamb * tf.abs(tf.angle(complex_W)))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class UnitaryReg(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        complex_W_conj_T = K.transpose(tf.conj(complex_W))
        I = tf.eye(tf.shape(complex_W)[0], dtype="complex64")

        return self.lamb * K.sum(K.abs(complex_W @ complex_W_conj_T - I))

    def get_config(self):
        return {'lamb': float(self.lamb),}

class UnitaryReg2(Regularizer):
    def __init__(self, lamb=0.1):
        self.lamb = K.cast_to_floatx(lamb)

    def __call__(self, W):
        complex_W = channels_to_complex(W)
        complex_W_conj_T = K.transpose(tf.conj(complex_W))
        I = tf.eye(tf.shape(complex_W)[1], dtype="complex64")

        return self.lamb * K.sum(K.abs(complex_W_conj_T @ complex_W - I))

    def get_config(self):
        return {'lamb': float(self.lamb),}


class InverseAndL2Reg(Regularizer):
    def __init__(self, l_u=0., l_a=0., otherW=None):
        self.l_u = K.cast_to_floatx(l_u)
        self.l_a = K.cast_to_floatx(l_a)
        self.otherW = otherW

    def __call__(self, W):
        regularization = 0.
        if self.l_u:
            complex_W = channels_to_complex(W)
            complex_T = channels_to_complex(self.otherW)
            I = tf.eye(tf.shape(complex_T)[0], dtype="complex64")
            regularization += K.sum(self.l_u * K.abs(complex_T @  complex_W  - I))
        if self.l_a:
            regularization += K.sum(self.l_a * K.square(W))
        return regularization

    def get_config(self):
        return {'l_u': float(self.l_u),
                'l_a': float(self.l_a),
                'otherW': self.otherW}

    
class RandomUnitaryNormal(keras.initializers.Initializer):
    """Initializer that generates tensors with a normal distribution.
    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self):
        return None

    def __call__(self, shape, dtype=None):
        import math as m
        def haar_measure(n):
            # generate a random unitary matrix
            z = (np.random.randn(n,n)+1j*np.random.randn(n,n))/m.sqrt(2.0)
            q,r = np.linalg.qr(z.astype('complex64'), mode='complete')
            d = np.diagonal(r)
            ph = d/np.absolute(d)
            q = q*ph 
            return q
        print('starting random unitary matrix generation of size %f\n',shape[0])
        w = haar_measure(shape[0])
        print('finishing random unitary matrix generation\n')
        return complex_to_channels(tf.complex(np.real(w), np.imag(w)))

    def get_config(self):
        return {}
    
    
class UnitaryAndL2Reg(Regularizer):
    def __init__(self, l_u=0., l_a=0.):
        self.l_u = K.cast_to_floatx(l_u)
        self.l_a = K.cast_to_floatx(l_a)

    def __call__(self, W):
        regularization = 0.
        if self.l_u:
            complex_W = channels_to_complex(W)
            complex_W_conj_T = K.transpose(tf.conj(complex_W))
            I1 = tf.eye(tf.shape(complex_W_conj_T)[0], dtype="complex64")
            #I2 = tf.eye(tf.shape(complex_W)[0], dtype="complex64")
            # check issues around W^TW (column orthonormal) vs WW^T (row orthonormal)
            regularization += K.sum(self.l_u *K.abs(complex_W_conj_T @ complex_W - I1))
                                    #+K.sum(self.l_u *K.abs(complex_W @ complex_W_conj_T - I2))
        if self.l_a:
            regularization += K.sum(self.l_a * K.square(W))
        return regularization

    def get_config(self):
        return {'l_u': float(self.l_u),
                'l_a': float(self.l_a)}


def amplitude_reg_l1(l=0.1):
    return AmplitudeRegL1(lamb=l)

def amplitude_reg_l2(l=0.1):
    return AmplitudeRegL2(lamb=l)

def phase_reg(l=0.1):
    return PhaseReg(lamb=l)

def unitary_reg(l=0.1):
    return UnitaryReg(lamb=l)

def unitary_reg_2(l=0.1):
    return UnitaryReg2(lamb=l)

def unitary_and_l2_reg(l_u=0.1, l_a=0.1):
    return UnitaryAndL2Reg(l_u, l_a)



## Layers
# Learnable Hadamard Product
class Hadamard(Layer):

    def __init__(self, **kwargs):
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        super(Hadamard, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, X):
        complex_X = channels_to_complex(X)
        complex_W = channels_to_complex(self.kernel)
        complex_res = complex_X @ complex_W    
        output = complex_to_channels(complex_res)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class GaussianPhaseDropout(Layer):
    """Apply multiplicative 1-centered Gaussian noise changing phase only
    As it is a regularization layer, it is only active at training time.
    The multiplicative noise will be uniform in 0-2pi
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, phase_range, **kwargs):
        super(GaussianPhaseDropout, self).__init__(**kwargs)
        self.phase_range = phase_range
        self.supports_masking = True

    def call(self, inputs, training=None):
        import math as m
        def noised():
            a = inputs         # no longer channels_to_complex(inputs)
            b = tf.random_uniform(shape=tf.shape(inputs)[0:2], 
                                  minval=-self.phase_range, maxval=self.phase_range, 
                                  #minval=-0.1*m.pi, maxval=0.1*m.pi,
                                  dtype=tf.float32, seed=None)
            return_val = complex_to_channels(tf.complex(a*tf.cos(b), a*tf.sin(b)))
            return return_val
        return K.in_train_phase(noised, complex_to_channels(tf.complex(inputs,0*inputs)), training=training)
    
    def build(self, input_shape):
        super(GaussianPhaseDropout, self).build(input_shape)

    def get_config(self):
        config = {'phase_range':self.phase_range}
        base_config = super(GaussianPhaseDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)

    
class ExpandtoChannels(Layer):
    """Expand to 2 channels for complex processing - inputs are amplitude vector
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self,  **kwargs):
        super(ExpandtoChannels, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return complex_to_channels(tf.complex(inputs,0*inputs))
    
    def build(self, input_shape):
        super(ExpandtoChannels, self).build(input_shape)

    def get_config(self):
        config = {}
        base_config = super(ExpandtoChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)
    
# Complex Dense Layer
class ComplexDense(Layer):

    def __init__(self, output_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim, 2),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      #constraint=self.kernel_constraint,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim, 2),
                                        initializer=self.bias_initializer,
                                        #regularizer=self.bias_regularizer,
                                        #constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        super(ComplexDense, self).build(input_shape)

    def call(self, X):
        # True Complex Multiplication (by channel combination)
        complex_X = channels_to_complex(X)
        complex_W = channels_to_complex(self.kernel)

        complex_res = complex_X @ complex_W
        
        if self.use_bias:
            complex_b = channels_to_complex(self.bias)
            complex_res = K.bias_add(complex_res, complex_b)
        
        output = complex_to_channels(complex_res)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, 2)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            #'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            #'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            #'kernel_constraint': constraints.serialize(self.kernel_constraint),
            #'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AmplitudeSq(Layer):

    def __init__(self, **kwargs):
        super(AmplitudeSq, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AmplitudeSq, self).build(input_shape)

    def call(self, X):
        complex_X = (channels_to_complex(X) if X.shape[-1] == 2 else X)
        output = tf.square(tf.abs(complex_X))  # take square of absolute output re physics of problem
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Amplitude(Layer):

    def __init__(self, **kwargs):
        super(Amplitude, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Amplitude, self).build(input_shape)

    def call(self, X):
        complex_X = (channels_to_complex(X) if X.shape[-1] == 2 else X)
        output = tf.abs(complex_X)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
class Phase(Layer):

    def __init__(self, **kwargs):
        super(Phase, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Phase, self).build(input_shape)

    def call(self, X):
        complex_X = (channels_to_complex(X) if X.shape[-1] == 2 else X)
        output = tf.angle(complex_X)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


### EXPERIMENTAL SECTION
# Hermitian Layer
class HermitianLayer(Layer):

    def __init__(self, output_dim,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        super(HermitianLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim, 2),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        # Called bias but really scalar multiplication of modes
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, 2),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        super(HermitianLayer, self).build(input_shape)

    def call(self, X):
        # True Complex Multiplication (by channel combination)
        complex_X = channels_to_complex(X)
        complex_W = channels_to_complex(self.kernel)
        complex_V = channels_to_complex(self.bias)
        complex_W_conj_T = tf.transpose(tf.conj(complex_W))

        complex_res = ((complex_X @ complex_W) * complex_V) @ complex_W_conj_T

        return complex_to_channels(complex_res)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(HermitianLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Contrast(Layer):

    def __init__(self, C_initial=10, **kwargs):
        super(Contrast, self).__init__(**kwargs)
        self.C_initial = C_initial

    def build(self, input_shape):
        self.C = self.add_weight(name='C', 
            shape=(),
            initializer=tf.initializers.random_uniform(-128, 128, seed=999),
            trainable=True)

    def call(self, X):
        F = (259 * (self.C + 255))/(255 * (259 - self.C))
        X_c = tf.clip_by_value(F * (X - 128) + 128, 0, 256)
        return X_c

    def compute_output_shape(self, input_shape):
        return input_shape

class Stack(Layer):

    def __init__(self, num_layers, **kwargs):
        super(Stack, self).__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        self.avg_weights = self.add_weight(name='weights',
            shape=(self.num_layers,),
            initializer = tf.initializers.constant(value=1/self.num_layers),
            trainable=True)

        super(Stack, self).build(input_shape)

    def call(self, X):
        stacked = tf.stack(X, axis=4)
        stacked = stacked * self.avg_weights
        return tf.reduce_mean(stacked, axis=4)


class DiagonalComplex(Layer):

    def __init__(self, output_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        super(DiagonalComplex, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, 2),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      #constraint=self.kernel_constraint,
                                      trainable=True)

        super(DiagonalComplex, self).build(input_shape)

    def call(self, X):
        # True Complex Multiplication (by channel combination)
        complex_X = channels_to_complex(X)
        complex_W = tf.diag(channels_to_complex(self.kernel))

        complex_res = complex_X @ complex_W

        output = complex_to_channels(complex_res)

        if self.activation is not None:
            output = self.activation(output)

        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, 2)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            #'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            #'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            #'kernel_constraint': constraints.serialize(self.kernel_constraint),
            #'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DiagonalComplex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class Hadamard1(Layer):

    def __init__(self, **kwargs):
        super(Hadamard1, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[1]),
                                      initializer=keras.initializers.Ones(),
                                      trainable=True)
        super(Hadamard1, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return complex_to_channels(channels_to_complex(x) * tf.complex(self.kernel,0*self.kernel))
    
    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(DiagonalComplex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],2)    
    
    
    
    
class Hadamard(Layer):

    def __init__(self, **kwargs):
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[1]),
                                      kernel_initializer=keras.initializers.Ones(),
                                      kernel_regularizer=self.kernel_regularizer,
                                      kernel_constraint=self.kernel_constraint,
                                      trainable=True)
        super(Hadamard, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print(x.shape, self.kernel.shape)
        return complex_to_channels(channels_to_complex(x) * self.kernel)
    
    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(DiagonalComplex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return input_shape
    
    
class DiagonalReal(Layer):

    def __init__(self, output_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        super(DiagonalReal, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, 2),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        super(DiagonalReal, self).build(input_shape)

    def call(self, X):
        # True Complex Multiplication (by channel combination)
        complex_X = channels_to_complex(X)
        complex_W = tf.diag(tf.abs(channels_to_complex(self.kernel)))

        complex_res = complex_X @ complex_W

        output = complex_to_channels(complex_res)

        if self.activation is not None:
            output = self.activation(output)

        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, 2)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            #'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            #'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            #'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            #'kernel_constraint': constraints.serialize(self.kernel_constraint),
            #'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DiagonalReal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

