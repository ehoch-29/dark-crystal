import math
import numpy as np
import scipy.special as spf
import scipy.interpolate as interp
import time
import quaternionic as qnc
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as clr
plt.rc('text', usetex=True)
import vsdm
from vsdm.units import *
from vsdm.utilities import *

import datetime as dts
from astropy.coordinates import SkyCoord, Galactic, ICRS
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

import earthspeed as ethp
from earthspeed import Windfinder

diffm, diffV = vsdm.testG_lm(5, -3, printout = False)
if np.abs(diffm) < 1e-14:
    print("testing 'spherical and WignerG: PASS")
else:
    print("testing 'spherical and WignerG: FAIL")
    print("check the version number for 'spherical', and fun testD_lm()")
print("vdsm", vsdm.__version__)
print("earthspeed", ethp.__version__)

deg = np.pi/180
unisize = 3.5

### Quaternion rotation
# finding the rotation that moves a 'z hat' unit vector to point towards (theta, phi)

def getQ(theta, phi):
    axisphi = phi + np.pi/2 #stationary under R
    axR = theta/2 
    qr = np.cos(axR)
    qi = np.sin(axR) * np.cos(axisphi)
    qj = np.sin(axR) * np.sin(axisphi)
    qk = 0. 
    return qnc.array(qr, qi, qj, qk)

def ix_thetaphi(theta, phi):
    return thetaphilist.index((theta, phi))

def getRcrl(Uvec, Fvec, gamma): 
    # U points up 
    # F points horizontally towards CCD
    # gamma is CCD poisition (on horizon) wrt north horizon
    Uhat = Uvec / np.linalg.norm(Uvec)
    zhat = np.array([0, 0, 1])
    nUvec = np.cross(Uhat, zhat) 
    if np.linalg.norm(nUvec) == 0:
        nUhat = np.array([1, 0, 0])
    else:
        nUhat = nUvec / np.linalg.norm(nUvec)
    alphaU = np.arccos(Uhat @ zhat)
    nU = qnc.array([0] + [qi for qi in nUhat])
    QrU = np.cos(0.5*alphaU) * qnc.array(1, 0, 0, 0) + np.sin(0.5*alphaU) * nU
    # act with qrU on U: returns zhat.
    # act with qrU on F:
    Fhat = Fvec / np.linalg.norm(Fvec)
    nF = qnc.array([0] + [qi for qi in Fhat])
    RuF = QrU * nF / QrU
    RuFhat = RuF.vector # just the imaginary part of RuF: a 3d vector
    # get azimuthal coordinate of RuF:
    is_one, thetaRuF, phiRuF = cart_to_sph(RuFhat)
    # difference between phiRuF and gamma: 
    alphaF = (gamma - phiRuF) % (2*np.pi) 
    # rotate RuF about +z axis to point towards CCD (at gamma):
    QrF = qnc.array(np.cos(0.5*alphaF), 0, 0, np.sin(0.5*alphaF))
    Qrcl = QrF * QrU
    return Qrcl

def rotateBy(qR, v):
    # rotate a vector v (or array of vectors) by rotation operator qR
    return qR * v / qR

def fRMS(R_list):
    Rbar = np.mean(R_list) 
    Rt = np.array(R_list)
    diff_sq = (Rt - Rbar)**2
    root_mean_sq = np.sqrt(np.mean(diff_sq))
    return root_mean_sq / Rbar

def chi2norm(R_list):
    Rbar = np.mean(R_list) 
    Rt = np.array(R_list)
    diff_sq = (Rt - Rbar)**2
    root_mean_sq = np.sqrt(np.mean(diff_sq))
    return root_mean_sq**2 / Rbar

### TRANS-STILBENE FORM FACTORS: s1, s2, s5, s8

"""Definition of the Q and V basis functions:"""
# (these should not be changed, without also changing McalK)

QMAX = 24*keV # Global value for q0=qMax for wavelets
Qbasis = dict(u0=QMAX, type='wavelet', uMax=QMAX)

VMAX = 820.*km_s # Global value for v0=vMax for wavelets
Vbasis = dict(u0=VMAX, type='wavelet', uMax=VMAX)

DeltaE = {}
DeltaE['s1'] = 4.24*eV
DeltaE['s2'] = 4.788*eV
DeltaE['s5'] = 5.791*eV
DeltaE['s8'] = 6.439*eV

"""When creating tstilbene_mK.hdf5, this format was used for the model labels:"""

ellMax = 24
nvMax = 511
nqMax = 1023
lmod = 2

def h5modelname(sI, dmModel, mX_MeV_float=0):
    """Define a consistent style for labeling McalI[dmModel] in this hdf5.

    returns: ('sI') + '/' + ('fdm') + '/' + ('mX_MeV')
        fdm = dmModel['fdm']
        mX_MeV = dmModel['mX']/MeV
    mX_MeV_float controls how many decimal points to include in the
        string version of mX/MeV
    """
    fdm = dmModel['fdm']
    mX_MeV = dmModel['mX'] / MeV
    if mX_MeV_float==0:
        mX_str = str(int(mX_MeV))
    elif mX_MeV_float==1:
        mX_str = '{:.1f}'.format(mX_MeV)
    elif mX_MeV_float==2:
        mX_str = '{:.12}'.format(mX_MeV)
    elif mX_MeV_float==3:
        mX_str = '{:.3f}'.format(mX_MeV)
    mIstr = '{}/{}/{}'.format(sI, fdm, mX_str)
    return mIstr

"""Import from HDF5"""

ellMax = 24
mXlist = [2, 3, 5, 10, 20, 30, 50, 100]

rates = {}
with h5py.File('tstilbene_mK.hdf5','r') as hdf5:
    for sI in ['s1', 's2', 's5', 's8']:
        for n in [0,2]:
            for mX in mXlist:
                dmModel = dict(mX=mX*MeV, fdm=n, mSM=mElec, DeltaE=DeltaE[sI])
                thislabel = h5modelname(sI, dmModel, mX_MeV_float=0)
                dataK = hdf5[thislabel+'/mcalK'][:]
                newK = vsdm.McalK(ellMax, lmod=2)
                newK.vecK = dataK
                rates[(sI, n, mX)] = newK


"""Combinations of s1, s2, s5, s8:"""

for mX in mXlist:
    for n in [0, 2]:
        vecK_s1 = rates[('s1', n, mX)].vecK
        vecK_tot = np.copy(vecK_s1)
        for sI in ['s2', 's5', 's8']:
            vecK_other = rates[(sI, n, mX)].vecK
            vecK_tot += vecK_other
        newK = vsdm.McalK(ellMax, lmod=2)
        newK.vecK = vecK_tot
        rates[('tot', n, mX)] = newK

"""Test: K^{(0)}_{00}, proportional to the isotropic average rate <R>"""
mX = mXlist[3]
n = 0
print('K_000 for mX = {} MeV, n = {}:'.format(mX, n))
for lbl in ['s1', 's2', 's5', 's8', 'tot']:
    K000 = rates[(lbl, n, mX)].vecK[0]
    print('{}:\t{}'.format(lbl, K000))

mX = mXlist[1]
n = 0
print('K_000 for mX = {} MeV, n = {}:'.format(mX, n))
for lbl in ['s1', 's2', 's5', 's8', 'tot']:
    K000 = rates[(lbl, n, mX)].vecK[0]
    print('{}:\t{}'.format(lbl, K000))

rotationlist = []
rotationarray = []
thetaphilist = []

for theta in range(0, 181, 2):
    rrow = []
    for phi in range(0, 361, 2):
        # for WignerG, want to apply the inverse of getQ.
        # getQ applied to vE moves vE from the z axis to the position (theta, phi). 
        # 1/getQ applied to the crystal does the same thing, 
        #     with vE -> (theta, phi) in the frame of the crystal.
        q = 1/getQ(theta * np.pi/180, phi * np.pi/180) 
        rrow += [q] 
        rotationlist += [q] 
        thetaphilist += [(theta, phi)]
    rotationarray += [rrow]

rotationarray = np.array(rotationarray)
# thetaphilist = np.array(thetaphilist)
# np.shape(rotationarray)
"""
# Calculate the Wigner G matrix for each rotation (takes ~2 minutes)

print('Calculating WignerG for {} orientations:'.format(len(rotationlist)))

t0 = time.time()
wG = vsdm.WignerG(ellMax, rotations=rotationlist, lmod=2)
tG = time.time() - t0
print('\t time: {:.2f} s'.format(tG))
"""



    
"""Convert this isotropic rate to events per second"""
mX_eg = 10
keylist = [('s1', 0, mX_eg), ('s1', 2, mX_eg), ('tot', 0, mX_eg), ('tot', 2, mX_eg)]
"""exposure factor for 1 kg of trans-stilbene, for 1 second"""
k0 = vsdm.g_k0(exp_kgyr=1./(3.15e7), mCell_g=721.0,
               sigma0_cm2=1e-38, rhoX_GeVcm3=0.4,
               v0=Vbasis['u0'], q0=Qbasis['u0'])
print(k0)
print('rates in Hz:')
for sI_m_n in keylist:
    K000 = rates[sI_m_n].vecK[0]
    print('{}: {}'.format(sI_m_n, K000*k0))
