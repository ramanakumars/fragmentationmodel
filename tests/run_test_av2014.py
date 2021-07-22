import numpy as np
from fragmodel.frag_funcs import FragModel
import matplotlib.pyplot as plt
import os 

## for testing cases
dir_path = os.path.dirname(os.path.realpath(__file__))

M      = 1.185e7
v      = 19040.
theta  = np.radians(17.)
h0     = 45000.
Ch     = 0.12/(2.e7)
sigma  = 0.3e6
alpha  = 0.18
Cfr    = 1.5
rho_d  = 3300.

fragmodel = FragModel(M, v, theta, h0, sigma, Ch, rho_d, Cfr, alpha)
fragmodel.Rp = 6371000.
fragmodel.g  = 9.81
fragmodel.Cd = 0.75
fragmodel.set_tp_prof(dir_path+'/earthtp/chelyabinsk_50k.tp',Ratmo=289.116)
fragmodel.add_fragment(0.025*M, 0.975e6, sigmafrag=10.5e6)

fragmodel.integrate(45.)

nfrags  = len(fragmodel.fraginfo["Mfrag"])
dEdt    = fragmodel.dEdtall
v       = fragmodel.v
theta   = fragmodel.theta
Edepo   = fragmodel.Edepo
h       = fragmodel.h

kt = 4.184e12

energydepdata = np.loadtxt(dir_path+'/energydep/ChelyabinskEnergyDep_Wheeler-et-al-2018.txt', skiprows=1)

colors = ['r', 'g']

fig1 = plt.figure()
ax1  = fig1.add_subplot(111)
ax1.plot(fragmodel.Edepoall*(1000./kt), h[:,0]/1000., 'k-', label='Modeled total deposited energy')
for i in range(nfrags):
    ax1.plot(fragmodel.Edepo[:,i]*(1000./kt), \
             fragmodel.h[:,i]/1000., '-', linewidth=0.5, color=colors[i], label='Fragment %d'%i)
ax1.plot(energydepdata[:,1], energydepdata[:,0], 'k--', label='Observed')
ax1.set_ylim((20., 45.))
ax1.set_xlabel(r'Energy deposited [kt/km]')
ax1.set_ylabel(r'Height [km]')
ax1.legend(loc='upper right')

fig2 = plt.figure()
ax2  = fig2.add_subplot(111)

## construct the ablated mass profile
mmain = fragmodel.M0*np.ones_like(fragmodel.dEdtall)

## discount mass loss from discrete fragmentation
for i in range(1,nfrags):
    mask = fragmodel.M[:,i] > 0
    mmain[mask] -= fragmodel.fraginfo['Mfrag'][i]
mainablation = 1. - fragmodel.M[:,0]/mmain

ax2.plot(mainablation, fragmodel.h[:,0]/1000., '-', color=colors[0], label='Main body')
for i in range(1, nfrags):
    mask = fragmodel.dEddt[:,i] > 0
    ax2.plot(1. - fragmodel.M[mask,i]/fragmodel.fraginfo['Mfrag'][i], \
             fragmodel.h[mask,i]/1000., '-', color=colors[i], label='Fragment %d'%i)
ax2.set_ylim((0., 45.))
ax2.set_xlim((0., 1.))
ax2.set_xlabel(r'Share of ablated mass')
ax2.set_ylabel(r'Height [km]')
ax2.legend(loc='upper right')

plt.show()
