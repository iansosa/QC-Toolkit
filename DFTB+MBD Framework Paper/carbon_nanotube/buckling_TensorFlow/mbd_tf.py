"""
This code is a modified version of Tensorflow implementation of Libmbd
See the original version: https://github.com/libmbd/libmbd/blob/master/src/pymbd/tensorflow.py

Additional functions are added:
1. Allow the hessian calculation using AD
2. Support different variants of MBD, i.e. MBD@rsSCS, MBD@SCS, MBD@TS
3. Support long-distance cut-off for the dipole-dipole interaction
"""

from math import pi, inf
import numpy as np
import tensorflow.compat.v1 as tf
from numpy.polynomial.legendre import leggauss

tf.disable_v2_behavior()
inf = float('inf')
pi = tf.constant(pi, tf.float64)
inf = tf.constant(inf, tf.float64)
ang = 1/0.529177249
class MBDEvaluator(object):
    def __init__(self, hessians=False, method='ts', **kwargs):
        self._inputs = coords, alpha_0, C6, R_vdw,beta = [
            tf.placeholder(tf.float64, shape=shape, name=name)
            for shape, name in [
                ((None, 3), 'coords'),
                ((None, ), 'alpha_0'),
                ((None, ), 'C6'),
                ((None, ), 'R_vdw'),
                ((), 'beta'),
            ]
        ]
        if method == 'ts':
            self._output = mbd_energy_ts(*self._inputs, **kwargs)
        elif method == 'scs':
            self._output = mbd_energy_scs(*self._inputs, **kwargs)
        elif method == 'rsscs':
            self._output = mbd_energy_rsscs(*self._inputs, **kwargs)
        else:
            raise ValueError('Unsupported mbd method')

        self._grad = tf.gradients(self._output, [self._inputs[0]])[0]

        if hessians:
            self._init_hessians()
        else:
            self._hessians = None

    def _init_hessians(self):
        self._hessians = tf.hessians(self._output, [self._inputs[0]])[0]

    def __call__(self, coords, alpha_0, C6, R_vdw, beta=1, hessians=None):
        inputs = dict(zip(self._inputs, [coords, alpha_0, C6, R_vdw, beta]))
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

def T_bare(R, R_mask=0):
    inf = tf.constant(np.inf, tf.float64)
    R_2 = tf.reduce_sum(R**2, -1)
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    if R_mask > 0:
        condition = tf.less(R_1, (R_mask*ang))
        R_1 = tf.where(condition, R_1, tf.ones_like(R_1)*100000)
    R_5 = _set_diag(R_1**5, inf)
    return (
        -3*R[:, :, :, None]*R[:, :, None, :]
        + R_2[:, :, None, None]*np.eye(3)[None, None, :, :]
    )/R_5[:,:,None,None]


def _set_diag(A, val):
    return tf.matrix_set_diag(A, tf.fill(tf.shape(A)[0:1], tf.cast(val, tf.float64)))

def freq_grid(n, L=0.6):
    x, w = leggauss(n)
    w = 2 * L / (1 - x) ** 2 * w
    x = L * (1 + x) / (1 - x)
    return np.hstack(([0], x[::-1])), np.hstack(([0], w[::-1]))

def _repeat(a, n):
    return tf.reshape(tf.tile(a[:, None], (1, n)), (-1,))


def mbd_energy_ts(coords, alpha_0, C6, R_vdw, beta, R_mask=0):
    pi = tf.constant(np.pi, tf.float64)
    omega = 4/3*C6/alpha_0**2
    sigma = (tf.sqrt(2/pi)*(alpha_0)/3)**(1/3)
    dipmat = dipole_matrix(coords, beta, R_vdw, sigma, R_mask, dip='erf')
    pre = _repeat(omega*tf.sqrt(alpha_0), 3)
    C = tf.diag(_repeat(omega**2, 3))+pre[:, None]*pre[None, :]*dipmat
    eigs, _ = tf.linalg.eigh(C)
    ene = tf.reduce_sum(tf.sqrt(tf.abs(eigs)))/2-3*tf.reduce_sum(omega)/2
    return ene

def mbd_energy_scs(coords, alpha_0, C6, R_vdw, beta, nfreq=15):
    pi = tf.constant(np.pi, tf.float64)
    freq, freq_w = freq_grid(nfreq)
    omega = 4 / 3 * C6 / alpha_0**2
    alpha_dyn = [alpha_0 / (1 + (u / omega) ** 2) for u in freq]
    alpha_dyn_rsscs = []
    for a in alpha_dyn:
        sigma = (tf.sqrt(2 / pi) * a / 3) ** (1 / 3)
        dipmat = dipole_matrix(
            coords, beta, sigma=sigma, R_vdw=R_vdw, dip='erf'
        )
        a_nlc = tf.linalg.inv(tf.diag(_repeat(1 / a, 3)) + dipmat)
        a_contr = sum(tf.reduce_sum(a_nlc[i::3, i::3], 1) for i in range(3)) / 3
        alpha_dyn_rsscs.append(a_contr)
    alpha_dyn_rsscs = tf.stack(alpha_dyn_rsscs)
    C6_rsscs = 3 / pi * tf.reduce_sum(freq_w[:, None] * alpha_dyn_rsscs**2, 0)
    R_vdw_rsscs = R_vdw * (alpha_dyn_rsscs[0, :] / alpha_0) ** (1 / 3)
    omega_rsscs = 4 / 3 * C6_rsscs / alpha_dyn_rsscs[0, :] ** 2
    sigma_rsscs = (tf.sqrt(2 / pi) * alpha_dyn_rsscs[0, :] / 3) ** (1 / 3)
    dipmat = dipole_matrix(coords, beta, sigma=sigma_rsscs, R_vdw=R_vdw_rsscs, dip='erf')
    pre = _repeat(omega_rsscs * tf.sqrt(alpha_dyn_rsscs[0, :]), 3)
    eigs, _ = tf.linalg.eigh(
        tf.diag(_repeat(omega_rsscs**2, 3)) + pre[:, None] * pre[None, :] * dipmat
    )
    ene = tf.reduce_sum(tf.sqrt(eigs)) / 2 - 3 * tf.reduce_sum(omega_rsscs) / 2
    return ene

def mbd_energy_rsscs(coords, alpha_0, C6, R_vdw, beta, nfreq=15):
    pi = tf.constant(np.pi, tf.float64)
    freq, freq_w = freq_grid(nfreq)
    omega = 4 / 3 * C6 / alpha_0**2
    alpha_dyn = [alpha_0 / (1 + (u / omega) ** 2) for u in freq]
    alpha_dyn_rsscs = []
    for a in alpha_dyn:
        sigma = (tf.sqrt(2 / pi) * a / 3) ** (1 / 3)
        dipmat = dipole_matrix(
            coords, beta, sigma=sigma, R_vdw=R_vdw, dip='erf,dip'
        )
        a_nlc = tf.linalg.inv(tf.diag(_repeat(1 / a, 3)) + dipmat)
        a_contr = sum(tf.reduce_sum(a_nlc[i::3, i::3], 1) for i in range(3)) / 3
        alpha_dyn_rsscs.append(a_contr)
    alpha_dyn_rsscs = tf.stack(alpha_dyn_rsscs)
    C6_rsscs = 3 / pi * tf.reduce_sum(freq_w[:, None] * alpha_dyn_rsscs**2, 0)
    R_vdw_rsscs = R_vdw * (alpha_dyn_rsscs[0, :] / alpha_0) ** (1 / 3)
    omega_rsscs = 4 / 3 * C6_rsscs / alpha_dyn_rsscs[0, :] ** 2
    dipmat = dipole_matrix(coords, beta, R_vdw=R_vdw_rsscs, dip='bare')
    pre = _repeat(omega_rsscs * tf.sqrt(alpha_dyn_rsscs[0, :]), 3)
    eigs, _ = tf.linalg.eigh(
        tf.diag(_repeat(omega_rsscs**2, 3)) + pre[:, None] * pre[None, :] * dipmat
    )
    ene = tf.reduce_sum(tf.sqrt(eigs)) / 2 - 3 * tf.reduce_sum(omega_rsscs) / 2
    return ene

def dipole_matrix(coords, beta, R_vdw=None, sigma=None, R_mask=0, dip=None):
    Rs = coords[:, None, :]-coords[None, :, :]
    if sigma is not None:
        sigmaij = tf.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
    S_vdw = 0.83 * (R_vdw[:, None] + R_vdw[None, :])
    dists = tf.sqrt(_set_diag(tf.reduce_sum(Rs ** 2, -1), 1e10))
    if dip == 'erf':
        dipmat = T_erf_coulomb(Rs, sigmaij, beta, R_mask)
    elif dip == 'bare':
        dipmat = damping_fermi(dists, S_vdw, 6)[:, :, None, None] * T_bare(Rs)
    elif dip == 'erf,dip':
        dipmat = (1 - damping_fermi(dists, S_vdw, 6)[:, :, None, None]) * T_erf_coulomb(
            Rs, sigmaij,beta)
    else:
        raise ValueError('Unsupported damping: {damping}')
    n_atoms = tf.shape(coords)[0]
    return tf.reshape(tf.transpose(dipmat, (0, 2, 1, 3)), (3*n_atoms, 3*n_atoms))

def T_erf_coulomb(R, sigma, beta, R_mask=0):
    pi = tf.constant(np.pi, tf.float64)
    inf = tf.constant(np.inf, tf.float64)
    bare = T_bare(R,R_mask)
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_1_inf = R_1
    R_1_0 = R_1
    if R_mask>0:
        condition = tf.less(R_1, (R_mask*ang))
        R_1_inf = tf.where(condition, R_1, tf.ones_like(R_1)*100000)
    R_5 = _set_diag(R_1_inf**5, inf)
    RR_R5 = R[:, :, :, None]*R[:, :, None, :]/R_5[:, :, None, None]
    zeta = R_1_0/(sigma*beta)
    theta = 2*zeta/tf.sqrt(pi)*tf.exp(-zeta**2)
    erf_theta = tf.erf(zeta) - theta
    return erf_theta[:, :, None, None]*bare + \
        (2*(zeta**2)*theta)[:, :, None, None]*RR_R5

