import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from fragmodel import frag_funcs
from fragmodel.frag_funcs import FragModel
import os 

## for testing cases
dir_path = os.path.dirname(os.path.realpath(__file__))

h  = 6.636e-34
k  = 1.381e-23
c  = 2.999e8
kt = 4.184e12

plt.rc("text",usetex=True)
plt.rc("text.latex",preamble=r"\usepackage{amsmath} \usepackage{gensymb} \usepackage{siunitx}")
# plt.rc("font",**{'family':'sans-serif'})
plt.rc("font",size="18")

def B_lam(lam, T):
    first  = (2.*h*c**2.)/(lam**5.)
    second = np.exp(h*c/(lam*k*T)) - 1.

    return first*1./second

def B_int(T):
    return (2*np.pi**4.*k**4.)/((15.*h**3.*c**2.))*(T**4.)

mass   = lambda r, rhod: (4./3.)*np.pi*rhod*(r**3.)

lam     = np.linspace(600., 700., 1000)*1.e-9
lam_lum = np.linspace(300., 900., 1000)*1.e-9

''' INITIAL VALUES '''
E0     = 111.8*kt
filter_tau = (4.85e12)/E0

'''  CREATE THE FIGURE TO HOLD THE CASES  '''
fig, axs = plt.subplots(3, 3, figsize=(14,12), sharex=True, sharey=True)
plt.subplots_adjust(top=0.96, right=0.96, left=0.08, bottom=0.08, hspace=0.15, wspace=0.11)

''' import the data '''
# corr     = 9.592
lcdata   = np.loadtxt(dir_path+"/lightcurve_hueso.csv", delimiter=",", skiprows=1)
lctime   = lcdata[:,1]
lcenergy = lcdata[:,3]/1.5
lcoffset = -lctime[lcenergy.argmax()]
totenghueso = np.trapz(lcenergy/(filter_tau), lctime)
### limit the times
lctrel      = lctime + lcoffset
lcresmask   = (lctrel>-0.6)&(lctrel<0.4)

print(lcenergy.max()/filter_tau)

''' CASE 1 - COMETARY 60 km/s '''
v      = 60000.
theta  = np.radians(65.)
h0     = 200000.
sigma_ab = 2.e-8
sigma  = 10.e3
alpha  = 0.0
Cfr    = 1.3
rho_d  = 500.
M      = 2.*E0/v**2.

frag_funcs.surface_area = lambda r: 2.*np.pi*(r**2.)

case1 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case1.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case1.add_fragment(0.02*M, 0.05e6, Cfr=3.,   alpha=0.0, sigmafrag=0.06e6)
case1.add_fragment(0.02*M, 0.06e6, Cfr=3.,   alpha=0.0, sigmafrag=0.10e6)
case1.add_fragment(0.02*M, 0.15e6, Cfr=2.0,  alpha=0.06)
case1.add_fragment(0.03*M, 0.21e6, Cfr=2.0,  alpha=0.00)
case1.add_fragment(0.04*M, 0.32e6, Cfr=3.,   alpha=0.02)
case1.add_fragment(0.03*M, 0.28e6, Cfr=3.,   alpha=0.06)
case1.add_fragment(0.06*M, 0.27e6, Cfr=1.2,  alpha=0.00, sigmafrag=0.12e6)
case1.add_fragment(0.22*M, 0.27e6, Cfr=1.8,  alpha=0.06, sigmafrag=0.20e6)
case1.add_fragment(0.06*M, 0.28e6, Cfr=1.8,  alpha=0.01, sigmafrag=0.58e6)
case1.integrate(tlim=10.)
nfrags  = len(case1.fraginfo["Mfrag"])
dtoff   = case1.t[case1.dEdtall.argmax()]
case1.t -= dtoff

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case1.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)

axs[0][0].plot(case1.t, case1.dEdtall, 'k-')
axs[0][0].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$=%.2f \sigma_E$"%(PerSigma))
axs[0][0].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[0][0].text(0.2,1.35e15, r"Case 1")

''' CASE 2 - STONY 60 km/s '''
v      = 60000.
theta  = np.radians(65.)
h0     = 200000.
sigma_ab = 0.04/(2e7)
sigma  = 0.5e6
alpha  = 0.03
Cfr    = 1.5
rho_d  = 2500.
M      = 2.*E0/v**2.

case2 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case2.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case2.add_fragment(0.05*M,  0.1e6,  Cfr=2.,    alpha=0.01, sigmafrag=0.25e6)
case2.add_fragment(0.04*M,  0.1e6,  Cfr=2.1,   alpha=0.01, sigmafrag=0.45e6)
case2.add_fragment(0.04*M,  0.7e6,  Cfr=2.2,   alpha=0.00)
case2.add_fragment(0.08*M,  0.75e6, Cfr=3.0,   alpha=0.05)# sigmafrag=0.35e6)
case2.add_fragment(0.05*M,  1.1e6, Cfr=1.8,   alpha=0.03)
case2.add_fragment(0.05*M,  1.6e6, Cfr=2.0,   alpha=0.00)
case2.add_fragment(0.06*M,  1.2e6,  Cfr=1.1,  alpha=0.00, sigmafrag=1.4e6)
case2.add_fragment(0.21*M,  1.2e6,  Cfr=2.0,  alpha=0.06, sigmafrag=1.55e6)
case2.add_fragment(0.04*M,  1.2e6,  Cfr=2.0,  alpha=0.01, sigmafrag=4.5e6)
case2.integrate(tlim=3.)
dtoff   = case2.t[case2.dEdtall.argmax()]
case2.t -= dtoff

nfrags  = len(case2.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case2.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)

axs[0][1].plot(case2.t, case2.dEdtall, 'k-')
axs[0][1].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
axs[0][1].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[0][1].text(0.2,1.35e15, r"Case 2")

''' CASE 3 - IRON 60 km/s '''
v      = 60000.
theta  = np.radians(65.)
h0     = 200000.
sigma_ab = 1.e-8
sigma  = 2.e6
alpha  = 0.03
Cfr    = 1.1
rho_d  = 5000.
M      = 2.*E0/v**2.

case3 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case3.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case3.add_fragment(0.03*M,  0.7e6, Cfr=2.,  alpha=0.03, sigmafrag=0.9e6)
case3.add_fragment(0.05*M,  0.9e6, Cfr=2.5, alpha=0.02, sigmafrag=1.5e6)
case3.add_fragment(0.05*M,  0.9e6, Cfr=3.0, alpha=0.03)#, sigmafrag=2.6e6)
case3.add_fragment(0.04*M,  3.2e6, Cfr=1.8, alpha=0.02)
case3.add_fragment(0.07*M,  3.7e6, Cfr=1.6, alpha=0.00)
case3.add_fragment(0.04*M,  4.9e6, Cfr=1.5, alpha=0.00)
case3.add_fragment(0.05*M,  4.4e6, Cfr=2.4, alpha=0.03, sigmafrag=5.3e6)
case3.add_fragment(0.21*M,  4.4e6, Cfr=2.2, alpha=0.07, sigmafrag=5.3e6)
case3.add_fragment(0.06*M,  4.6e6, Cfr=2.4, alpha=0.02, sigmafrag=12.5e6)
case3.integrate(tlim=10.)
dtoff   = case3.t[case3.dEdtall.argmax()]
case3.t -= dtoff

nfrags  = len(case3.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case3.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)


axs[0][2].plot(case3.t, case3.dEdtall, 'k-')
axs[0][2].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$ = %.2f \sigma_E $"%(PerSigma))
axs[0][2].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[0][2].text(0.2,1.35e15, r"Case 3")

''' CASE 4 - COMETARY 65 km/s '''
v      = 65000.
theta  = np.radians(50.)
h0     = 200000.
sigma_ab = 2e-8
sigma  = 10.e3
alpha  = 0.01
Cfr    = 1.3
rho_d  = 500.
M      = 2.*E0/v**2.

case4 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case4.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case4.add_fragment(0.02*M, 0.05e6, Cfr=3.,   alpha=0.0, sigmafrag=0.06e6)
case4.add_fragment(0.02*M, 0.06e6, Cfr=3.,   alpha=0.0, sigmafrag=0.10e6)
case4.add_fragment(0.02*M, 0.15e6, Cfr=2.0,  alpha=0.06)
case4.add_fragment(0.03*M, 0.21e6, Cfr=2.0,  alpha=0.00)
case4.add_fragment(0.04*M, 0.32e6, Cfr=3.,   alpha=0.01)
case4.add_fragment(0.02*M, 0.28e6, Cfr=3.,   alpha=0.04)
case4.add_fragment(0.06*M, 0.27e6, Cfr=1.2,  alpha=0.00, sigmafrag=0.08e6)
case4.add_fragment(0.22*M, 0.27e6, Cfr=1.8,  alpha=0.06, sigmafrag=0.17e6)
case4.add_fragment(0.05*M, 0.28e6, Cfr=1.8,  alpha=0.01, sigmafrag=0.52e6)

case4.integrate(tlim=10.)
dtoff   = case4.t[case4.dEdtall.argmax()]
case4.t -= dtoff

nfrags  = len(case4.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case4.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)


axs[1][0].plot(case4.t, case4.dEdtall, 'k-')
axs[1][0].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$ = %.2f \sigma_E $"%(PerSigma))
axs[1][0].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[1][0].text(0.2,1.35e15, r"Case 4")

''' CASE 5 - STONY 65 km/s '''
v      = 65000.
theta  = np.radians(50.)
h0     = 200000.
sigma_ab = 0.04/(2e7)
sigma  = 0.2e6
alpha  = 0.03
Cfr    = 1.5
rho_d  = 2500.
M      = 2.*E0/v**2.

case5 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case5.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case5.add_fragment(0.05*M,  0.05e6,  Cfr=2.,    alpha=0.01, sigmafrag=0.15e6)
case5.add_fragment(0.04*M,  0.45e6,  Cfr=2.1,   alpha=0.0,  sigmafrag=0.02e6)
case5.add_fragment(0.05*M,  0.45e6,  Cfr=2.0,   alpha=0.00)
case5.add_fragment(0.06*M,  0.55e6,  Cfr=2.0,   alpha=0.00)# sigmafrag=0.35e6)
case5.add_fragment(0.02*M,  0.85e6,  Cfr=1.8,   alpha=0.00)
case5.add_fragment(0.04*M,  0.85e6,  Cfr=1.8,   alpha=0.03)
case5.add_fragment(0.065*M, 0.7e6,   Cfr=1.1,   alpha=0.00, sigmafrag=0.7e6)
case5.add_fragment(0.21*M,  0.7e6,   Cfr=2.0,   alpha=0.06, sigmafrag=0.8e6)
case5.add_fragment(0.04*M,  0.7e6,   Cfr=2.0,   alpha=0.01, sigmafrag=2.3e6)

case5.integrate(tlim=3.)
dtoff   = case5.t[case5.dEdtall.argmax()]
case5.t -= dtoff

nfrags  = len(case5.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case5.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)

axs[1][1].plot(case5.t, case5.dEdtall, 'k-')
axs[1][1].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$ = %.2f \sigma_E $"%(PerSigma))
axs[1][1].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[1][1].text(0.2,1.35e15, r"Case 5")

''' CASE 6 - IRON 65 km/s '''
v          = 65000.
theta      = np.radians(50.)
h0         = 200000.
sigma_ab   = 1.e-8
sigma      = 2.e6
alpha      = 0.05
Cfr        = 1.1
rho_d      = 5000.
M          = 2.*E0/v**2.

case6 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case6.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case6.add_fragment(0.02*M,  0.8e6,  Cfr=2.,  alpha=0.03, sigmafrag=1.1e6)
case6.add_fragment(0.05*M,  1.5e6,  Cfr=2.,  alpha=0.02, sigmafrag=1.5e6)
case6.add_fragment(0.03*M,  0.7e6,  Cfr=3.2, alpha=0.00)#, sigmafrag=2.6e6)
case6.add_fragment(0.05*M,  1.2e6,  Cfr=2.8, alpha=0.03)
case6.add_fragment(0.03*M,  3.6e6,  Cfr=2.8, alpha=0.03)
case6.add_fragment(0.04*M,  3.6e6,  Cfr=1.8, alpha=0.05)
case6.add_fragment(0.03*M,  4.1e6,  Cfr=1.8, alpha=0.00)
case6.add_fragment(0.05*M,  4.9e6,  Cfr=2.0, alpha=0.00)
case6.add_fragment(0.04*M,  4.5e6,  Cfr=2.4, alpha=0.03, sigmafrag=5.1e6)
case6.add_fragment(0.21*M,  4.5e6,  Cfr=2.2, alpha=0.06, sigmafrag=5.1e6)
case6.add_fragment(0.05*M,  5.5e6,  Cfr=2.4, alpha=0.02, sigmafrag=10.4e6)
case6.integrate(tlim=10.)
dtoff   = case6.t[case6.dEdtall.argmax()]
case6.t -= dtoff

nfrags  = len(case6.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case6.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)

axs[1][2].plot(case6.t, case6.dEdtall, 'k-')
axs[1][2].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$ = %.2f \sigma_E $"%(PerSigma))
axs[1][2].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[1][2].text(0.2,1.35e15, r"Case 6")

''' CASE 7 - COMETARY 70 km/s '''
v      = 70000.
theta  = np.radians(50.)
h0     = 200000.
sigma_ab = 2e-8
sigma  = 10.e3
alpha  = 0.01
Cfr    = 1.3
rho_d  = 500.
M      = 2.*E0/v**2.

case7 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case7.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case7.add_fragment(0.02*M,  0.05e6, Cfr=3.,   alpha=0.0, sigmafrag=0.06e6)
case7.add_fragment(0.02*M,  0.06e6, Cfr=3.,   alpha=0.0, sigmafrag=0.10e6)
case7.add_fragment(0.02*M,  0.15e6, Cfr=2.0,  alpha=0.06)
case7.add_fragment(0.03*M,  0.21e6, Cfr=2.0,  alpha=0.00)
case7.add_fragment(0.04*M,  0.32e6, Cfr=3.,   alpha=0.01)
case7.add_fragment(0.025*M, 0.28e6, Cfr=3.1,  alpha=0.04)
case7.add_fragment(0.05*M,  0.27e6, Cfr=1.2,  alpha=0.00, sigmafrag=0.16e6)
case7.add_fragment(0.23*M,  0.27e6, Cfr=1.8,  alpha=0.06, sigmafrag=0.21e6)
case7.add_fragment(0.05*M,  0.28e6, Cfr=1.8,  alpha=0.01, sigmafrag=0.57e6)
case7.integrate(tlim=10.)

dtoff   = case7.t[case7.dEdtall.argmax()]
case7.t -= dtoff
nfrags  = len(case7.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case7.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)

axs[2][0].plot(case7.t, case7.dEdtall, 'k-')
axs[2][0].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$ = %.2f \sigma_E $"%(PerSigma))
axs[2][0].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[2][0].text(0.2,1.35e15, r"Case 7")

''' CASE 8 - STONY 70 km/s '''
v      = 70000.
theta  = np.radians(50.)
h0     = 200000.
sigma_ab = 2.e-9
sigma  = 0.5e6
alpha  = 0.03
Cfr    = 1.5
rho_d  = 2500.
M      = 2.*E0/v**2.

case8 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case8.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case8.add_fragment(0.05*M,  0.10e6,  Cfr=2.,   alpha=0.01, sigmafrag=0.24e6)
case8.add_fragment(0.05*M,  0.12e6,  Cfr=2.1,  alpha=0.0,  sigmafrag=0.46e6)
case8.add_fragment(0.04*M,  0.52e6,  Cfr=2.0,  alpha=0.00)
case8.add_fragment(0.06*M,  0.8e6,   Cfr=2.0,  alpha=0.00)# sigmafrag=0.35e6)
case8.add_fragment(0.055*M, 1.09e6,  Cfr=2.0,  alpha=0.03)
case8.add_fragment(0.04*M,  1.45e6,  Cfr=1.9,  alpha=0.00)
case8.add_fragment(0.06*M,  0.9e6,   Cfr=1.1,  alpha=0.00, sigmafrag=1.35e6)
case8.add_fragment(0.22*M,  0.9e6,   Cfr=2.0,  alpha=0.06, sigmafrag=1.4e6)
case8.add_fragment(0.04*M,  0.9e6,   Cfr=2.0,  alpha=0.01, sigmafrag=3.9e6)
case8.integrate(tlim=3.)

dtoff   = case8.t[case8.dEdtall.argmax()]
case8.t -= dtoff
nfrags  = len(case8.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case8.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)

axs[2][1].plot(case8.t, case8.dEdtall, 'k-')
axs[2][1].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$ = %.2f \sigma_E $"%(PerSigma))
axs[2][1].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f} $ kt"%(AvgRes/kt))
axs[2][1].text(0.2,1.35e15, r"Case 8")

''' CASE 9 - IRON 70 km/s '''
v        = 70000.
theta    = np.radians(45.)
h0       = 200000.
sigma_ab = 1.e-8
sigma    = 5.e6
alpha    = 0.05
Cfr      = 1.1
rho_d    = 5000.
M        = 2.*E0/v**2.

case9 = FragModel(M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha)
case9.set_tp_prof(dir_path+'/juptp/jupiter.tp',Ratmo=3637.)

''' add fragments as (mass, Prelease, [Cfr, alpha, sigma]) '''
case9.add_fragment(0.01*M,  1.3e6,  Cfr=2.,  alpha=0.00, sigmafrag=2.8e6)
case9.add_fragment(0.03*M,  2.7e6,  Cfr=2.5, alpha=0.00, sigmafrag=3.7e6)
case9.add_fragment(0.02*M,  6.0e6,  Cfr=3.0, alpha=0.02)#, sigmafrag=2.6e6)
case9.add_fragment(0.03*M,  5.9e6,  Cfr=1.4, alpha=0.01)
case9.add_fragment(0.04*M,  6.5e6,  Cfr=2.4, alpha=0.01, sigmafrag=11.0e6)
case9.add_fragment(0.24*M,  6.5e6,  Cfr=2.4, alpha=0.05, sigmafrag=11.2e6)
case9.add_fragment(0.09*M,  6.9e6,  Cfr=2.1, alpha=0.00, sigmafrag=23.4e6)
case9.integrate(tlim=10.)

dtoff   = case9.t[case9.dEdtall.argmax()]
case9.t -= dtoff
nfrags  = len(case9.fraginfo["Mfrag"])

# print("Total energy from lc: %.3f kt"%(totenghueso/kt))

''' interpolate the data onto the observed time steps to calculate res '''
Eint    = 10.**case9.Et(lctime+lcoffset)
Eres    = lcenergy/filter_tau - Eint
Ereslim = Eres[lcresmask]#/1.e13

''' calculate R^2 '''
SStot  = np.sum((Eint[lcresmask]-Eint[lcresmask].mean())**2.)
SSres  = np.sum(Ereslim**2.)
Rsq    = 1. - SSres/SStot
AvgRes = np.abs(Ereslim).mean()
PerSigma = AvgRes/(15.*kt)

axs[2][2].plot(case9.t, case9.dEdtall, 'k-')
axs[2][2].plot(lctime+lcoffset, lcenergy/filter_tau, 'r-')
#axs\[([0-9])\]\[([0-9])\].text\(-0.43, 1.2e15, r"$ = %.2f \sigma_E $"%(PerSigma))
axs[2][2].text(-0.55, 1.35e15, r"$\Delta E=\num{%.1f}$ kt"%(AvgRes/kt))
axs[2][2].text(0.2,1.35e15, r"Case 9")

axs[0][0].set_ylim((0., 1.5e15))
axs[0][0].set_xlim((-0.616, 0.4))

axs[0][0].set_title('Cometary')
axs[0][1].set_title('Stony')
axs[0][2].set_title('Iron-nickel')

rightax1 = axs[0][2].twinx()
rightax2 = axs[1][2].twinx()
rightax3 = axs[2][2].twinx()

rightax1.set_ylabel(r'$v=60$ km/s', labelpad=10)
rightax2.set_ylabel(r'$v=65$ km/s', labelpad=10)
rightax3.set_ylabel(r'$v=70$ km/s', labelpad=10)

rightax1.set_yticks([])
rightax2.set_yticks([])
rightax3.set_yticks([])


axs[0][0].set_ylabel(r'$dE/dt$ [W]')
axs[1][0].set_ylabel(r'$dE/dt$ [W]')
axs[2][0].set_ylabel(r'$dE/dt$ [W]')

axs[2][0].set_xlabel(r'Time [s]')
axs[2][1].set_xlabel(r'Time [s]')
axs[2][2].set_xlabel(r'Time [s]')


plt.show()
