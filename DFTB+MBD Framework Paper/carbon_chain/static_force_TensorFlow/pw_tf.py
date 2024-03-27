from __future__ import division, print_function
from math import pi, inf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

inf = float('inf')
pi = tf.constant(pi, tf.float64)
inf = tf.constant(inf, tf.float64)
ang = 1/0.529177249

class PWEvaluator(object):
    def __init__(self, hessians=False, model='LJ', **kwargs):
        self._inputs = coords, eps, sigma,  alpha_0, C6,  R_vdw = [
            tf.placeholder(tf.float64, shape=shape, name=name)
            for shape, name in [
                ((None, 3), 'coords'),
                ((None, ), 'eps'),
                ((None, ), 'sigma'),
                ((None,), 'alpha_0'),
                ((None,), 'C6'),
                ((None, ), 'R_vdw')
            ]
        ]
        if model == 'damped':
            self._output = pw_energy_damped(*self._inputs, **kwargs)
        elif model == 'LJ':
            self._output = pw_energy_LJ(*self._inputs, **kwargs)
        elif model == 'TS':
            self._output = pw_energy_TS(*self._inputs, **kwargs)
        else:
            raise ValueError('Unsupported mbd method')

        self._grad = tf.gradients(self._output, [self._inputs[0]])[0]

        if hessians:
            self._init_hessians()
        else:
            self._hessians = None

    def _init_hessians(self):
        self._hessians = tf.hessians(self._output, [self._inputs[0]])[0]


    def __call__(self, coords, eps, sigma,  alpha_0, C6,  R_vdw,  hessians=None):
        inputs = dict(zip(self._inputs, [coords, eps, sigma, alpha_0, C6, R_vdw]))
        outputs = self._output, self._grad
        if hessians or hessians is None and self._hessians is not None:
            if self._hessians is None:
                self._init_hessians()
            outputs = self._hessians
        with tf.Session() as sess:
            out = sess.run(outputs, inputs)
            return out

def damping_fermi(R, S_vdw, d):
    return 1/(1+tf.exp(-d*(R/S_vdw-1)))
def damping_f(R, S_vdw, d):
    return (1-tf.exp(-0.076*(R/0.96))**7)**7

def T_bare(R, eps, sigma):
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_6 = _set_diag(R_1**6, 1e10)
    epsij   = tf.sqrt((eps[:, None]*eps[None, :]))
    sigmaij = 0.5*(sigma[:, None]+sigma[None, :])
    return -4*epsij[:,:,None,None]*sigmaij[:,:,None,None]**6/R_6[:,:,None,None]

def _set_diag(A, val):
    return tf.matrix_set_diag(A, tf.fill(tf.shape(A)[0:1], tf.cast(val, tf.float64)))

def _repeat(a, n):
    return tf.reshape(tf.tile(a[:, None], (1, n)), (-1,))

def pw_energy_damped(coords, eps, sigma, alpha_0, C6, R_vdw):
    Rs = coords[:, None, :]-coords[None, :, :]
    dists = tf.sqrt(_set_diag(tf.reduce_sum(Rs**2, -1), 1e10))
    S_vdw = 1.0*(R_vdw[:, None]+R_vdw[None, :])
    ene  = damping_f(dists, S_vdw, 0)[:, :, None, None]*T_bare(Rs, eps, sigma)
    return tf.reduce_sum(ene)/2

def pw_energy_LJ(coords, eps, sigma, alpha_0, C6, R_vdw):
    R = coords[:, None, :] - coords[None, :, :]
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_6 = _set_diag(R_1**6, 1e10)
    R_12 = _set_diag(R_1 ** 12, 1e10)
    epsij   = tf.sqrt((eps[:, None]*eps[None, :]))
    sigmaij = 0.5*(sigma[:, None]+sigma[None, :])
    ene = 4*epsij[:,:,None,None]*(sigmaij[:,:,None,None]**12/R_12[:,:,None,None]-sigmaij[:,:,None,None]**6/R_6[:,:,None,None])
    return tf.reduce_sum(ene)/2

def pw_energy_TS(coords, eps, sigma, alpha_0, C6, R_vdw):
    Rs = coords[:, None, :]-coords[None, :, :]
    S_vdw = 0.94*(R_vdw[:, None]+R_vdw[None, :])
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(Rs**2, -1), 1e-10))
    R_6 = _set_diag(R_1**6, 1e10)
    C6ij = 2*C6[:, None]*C6[None, :]/((alpha_0[:, None]/alpha_0[None, :])*C6[None,:]+(alpha_0[None,:]/alpha_0[:,None])*C6[:, None])
    ene  = -damping_fermi(R_1, S_vdw, 20.)[:, :, None, None]*C6ij[:,:,None,None]/R_6[:,:,None,None]
    return tf.reduce_sum(ene)/2
