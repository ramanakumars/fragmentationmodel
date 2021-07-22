import numpy as np
from scipy.interpolate import interp1d

surface_area = lambda r: np.pi*(r**2.)

class FragModel():
    '''
        Fragmentation model based on Avramenko et al. 2014 (Av 2014)
    '''
    def __init__(self, M, v, theta, h0, sigma, sigma_ab, rho_d, Cfr, alpha):
        ''' initialize values '''
        self.M0       = M     ## initial mass
        self.v0       = v     ## initial velocity
        self.theta0   = theta ## angle wrt horizontal
        self.h0       = h0    ## initial height
        self.s0       = sigma ## initial material strength
        self.sigma_ab = sigma_ab    ## ablation coefficient
        self.rho_d    = rho_d ## meteor density
        self.r0       = (M/((4./3.)*np.pi*rho_d))**(1./3.) ## radius of meteor
        self.S0       = surface_area(self.r0)  ## cross-sectional area
        self.Cfr      = Cfr   ## fragmentation coefficient
        self.alpha    = alpha ## strength power scale

        ''' constants '''
        self.Qa    = 1.e7  ## heat of ablation J/kg 
        self.dt    = 5.e-4 ## timestep in seconds
        self.Cr    = 0.37  ## ratio of ablation released as heat (Av 2014)
        self.Cd    = 0.92  ## from Carter, Jandir & Kress results in 2009 LPSC
        self.g     = 24.00  ## gravity
        self.Rp    = 70000000 ## planet radius

        print("Initializing with diameter %.3f"%(self.r0*2))

        ''' create the data structures to hold results '''
        self.out   = {}
        self.t       = []
        self.M       = []
        self.v       = []
        self.r       = []
        self.S       = []
        self.h       = []
        self.theta   = []
        self.DynPres = []
        self.sigma   = []
        self.Mfr     = []
        self.Nfr     = []
        self.dErdt   = []
        self.dEddt   = []

        ''' set up the dictionary to calculate frags '''
        self.fraginfo              = {}
        self.fraginfo["Mfrag"]     = []
        self.fraginfo["sigmafrag"] = []
        self.fraginfo["Prelease"]  = []
        self.fraginfo["Cfr"]       = []
        self.fraginfo["alpha"]     = []
        ## track if the fragment has been released
        self.fraginfo["released"]  = []
        self.fraginfo["done"]      = []

        ''' 
            the main body is released 
            this just helps with bookkeeping
        ''' 
        self.fraginfo["Mfrag"].append(M)
        self.fraginfo["sigmafrag"].append(sigma)
        self.fraginfo["Prelease"].append(0.)
        self.fraginfo["Cfr"].append(Cfr)
        self.fraginfo["alpha"].append(alpha)

        self.fraginfo["released"].append(True)
        self.fraginfo["done"].append(False)

    def set_tp_prof(self, tpzfile, Ratmo=3637.):
        ## Get TP profile
        tpz = np.genfromtxt(tpzfile,skip_header=1,delimiter=',')

        ## Interpolate the tpz profile
        zd = tpz[:,0] ## in km
        Pd = tpz[:,1] ## in mbar
        Td = tpz[:,2] ## in K

        Pz = interp1d(zd*1000., np.log10(Pd), kind='cubic')
        Tz = interp1d(zd*1000., Td, kind='cubic')

        self.rhoz = lambda z: 10.**(Pz(z)+2.)/(Ratmo*Tz(z))
        self.Pz   = lambda z: 10.**(Pz(z)+2.)

    def add_fragment(self, M, Prelease, Cfr=1.5, alpha=0.0, sigmafrag=-1):
        self.fraginfo["Mfrag"].append(M)
        self.fraginfo["Prelease"].append(Prelease)
        self.fraginfo["sigmafrag"].append(sigmafrag)
        self.fraginfo["Cfr"].append(Cfr)
        self.fraginfo["alpha"].append(alpha)

        self.fraginfo["released"].append(False)
        self.fraginfo["done"].append(False)

        # if(sigmafrag != -1):
        #     print("%.2f & %.2f & %.1f & %.2f & %.2f \\\\"%(M/1000., Prelease/1.e6, Cfr, alpha, sigmafrag/1.e6))
        # else:
        #     print("%.2f & %.2f & %.1f & %.2f & - \\\\"%(M/1000., Prelease/1.e6, Cfr, alpha))

    def integrate(self, tlim=20.0, vlim=100., hlim=100., mlim=0.05, verbose=1, offset_time=True):
        Qa  = self.Qa
        Cr  = self.Cr
        dt  = self.dt
        Cd  = self.Cd 
        Cfr = self.Cfr
        g   = self.g
        Rp  = self.Rp

        t     = 0.
        M     = self.M0
        v     = self.v0
        theta = self.theta0
        h     = self.h0
        r     = self.r0
        S     = self.S0
        s0    = self.s0
        Nfr   = 1
        Mfr   = M
        sigma = s0
        rho_d = self.rho_d
        rho_a = self.rhoz(h)
        Pram  = rho_a*(v**2.)
        alpha = self.alpha
        sigma_ab = self.sigma_ab


        nfrags = len(self.fraginfo["Mfrag"])

        self.M.append(np.zeros(nfrags))
        self.v.append(np.zeros(nfrags))
        self.r.append(np.zeros(nfrags))
        self.S.append(np.zeros(nfrags))
        self.h.append(np.zeros(nfrags))
        self.theta.append(np.zeros(nfrags))
        self.DynPres.append(np.zeros(nfrags))
        self.sigma.append(np.zeros(nfrags))
        self.Mfr.append(np.zeros(nfrags))
        self.Nfr.append(np.zeros(nfrags))
        self.dErdt.append(np.zeros(nfrags))
        self.dEddt.append(np.zeros(nfrags))


        ''' tracking the beginning and end of fragmentation '''
        self.trelease = np.zeros(nfrags)
        self.hrelease = np.zeros(nfrags)
        self.vrelease = np.zeros(nfrags)
        
        self.tend     = np.zeros(nfrags)
        self.hend     = np.zeros(nfrags)
        self.vend     = np.zeros(nfrags)
        self.mend     = np.zeros(nfrags)

        self.t.append(t)
        self.M[0][0]       = M
        self.v[0][0]       = v
        self.S[0][0]       = S
        self.r[0][0]       = r
        self.h[0][:]       = h
        self.theta[0][0]   = theta
        self.DynPres[0][0] = Pram
        self.sigma[0][0]   = sigma
        self.Mfr[0][0]     = Mfr
        self.Nfr[0][0]     = Nfr
        self.dErdt[0][0]   = 0.
        self.dEddt[0][0]   = 0.

        end = False

        while(end == False):
            self.M.append(np.zeros(nfrags))
            self.v.append(np.zeros(nfrags))
            self.r.append(np.zeros(nfrags))
            self.S.append(np.zeros(nfrags))
            self.h.append(np.zeros(nfrags))
            self.theta.append(np.zeros(nfrags))
            self.DynPres.append(np.zeros(nfrags))
            self.sigma.append(np.zeros(nfrags))
            self.Mfr.append(np.zeros(nfrags))
            self.Nfr.append(np.zeros(nfrags))
            self.dErdt.append(np.zeros(nfrags))
            self.dEddt.append(np.zeros(nfrags))


            for i in range(nfrags):
                ''' get the motion of each fragment '''
                if((self.fraginfo["released"][i])& \
                    (not self.fraginfo["done"][i])):
                    '''
                        get the fragment information in the previous
                        timestep
                    '''
                    v     = self.v[-2][i]
                    M     = self.M[-2][i]
                    S     = self.S[-2][i]
                    h     = self.h[-2][i]
                    theta = self.theta[-2][i]
                    sigma = self.sigma[-2][i]
                    Nfr   = self.Nfr[-2][i]
                    Mfr   = self.Mfr[-2][i]

                    rho_a = self.rhoz(h)
                    Pram  = rho_a*(v**2.)

                    dvdt  = -Cd*S*rho_a*(v**2.)/(2.*M) + g*np.sin(theta)
                    dMdt  = -sigma_ab*S*rho_a*(np.abs(v)**3.)

                    dthetadt  = g*np.cos(theta)/v - v*np.sin(theta)/(Rp + h)

                    dEdtd = Cd*S*rho_a*(np.abs(v)**3.)/2.
                    dEdta = sigma_ab*S*rho_a*(np.abs(v)**5.)/(2.)

                    dErdt = dEdtd + dEdta    ## released energy
                    dEddt = dErdt - Cr*dEdta ## deposited energy
                    dSdt  = (2./3.)*(S/M)*dMdt
                    
                    if(Pram > sigma):
                        Cfr    = self.fraginfo["Cfr"][i]
                        s0     = self.fraginfo['sigmafrag'][i]
                        M0     = self.fraginfo['Mfrag'][i]
                        alpha  = self.fraginfo["alpha"][i]

                        dSdtfr = Cfr*np.sqrt(Pram - sigma)/(M**(1./3.)*rho_d**(1./6.))*S
                        dSdt  += dSdtfr

                        Nfr    = 16.*(S**3.)*rho_d**2./(9.*np.pi*M**2.)
                        Mfr    = M/Nfr
                        sigma  = s0*(M0/Mfr)**(alpha) 

                    
                    M  += dMdt*dt
                    S  += dSdt*dt
                    theta  += dthetadt*dt
                    h  += -v*np.sin(theta)*dt
                    v  += dvdt*dt

                    ''' fill in the stuff for the fragment '''
                    self.M[-1][i]       = M
                    self.v[-1][i]       = v
                    self.theta[-1][i]   = theta
                    self.r[-1][i]       = np.sqrt(S/(np.pi))
                    self.S[-1][i]       = S
                    self.h[-1][i]       = h
                    self.DynPres[-1][i] = Pram
                    self.sigma[-1][i]   = sigma
                    self.Mfr[-1][i]     = Mfr
                    self.Nfr[-1][i]     = Nfr
                    self.dErdt[-1][i]   = dErdt
                    self.dEddt[-1][i]   = dEddt

                    '''
                        check if we need to stop working this fragment
                    '''
                    ### stop conditions
                    if((self.v[-1][i] < vlim)|(self.M[-1][i] < mlim)|(self.S[-1][i]<0.)):
                        self.fraginfo["done"][i] = True
                        
                        self.hend[i]  = h
                        self.tend[i]  = t
                        self.vend[i]  = v
                        self.mend[i]  = M



                ''' set the height of the fragment to be the 
                    same as the main body
                    just for bookkeeping purposes
                '''
                if(not self.fraginfo['released'][i]):
                    self.h[-1][i] = self.h[-1][0]
            ''' 
                check if the body needs to fragmented.
                fragment release happens at the next 
                timestep 
            '''
            for i in range(1, nfrags):
                if((self.DynPres[-1][0] >= self.fraginfo["Prelease"][i])& \
                    (not self.fraginfo["released"][i])):
                        Mfrag     = self.fraginfo["Mfrag"][i]
                        sigmafrag = self.fraginfo["sigmafrag"][i]
                        
                        self.fraginfo["released"][i] = True

                        ''' remove mass and surface area from the main body '''
                        Mmain                 = self.M[-1][0]
                        self.M[-1][0] -= Mfrag
                        
                        ''' 
                            assume that the fragment is a sphere 
                            probably true considering these are 'small' fragments
                        '''
                        
                        rfrag = (Mfrag/((4./3.)*np.pi*rho_d))**(1./3.)

                        ''' 
                            recalculate the new radius for the main body
                            by assuming 
                            r(M) = r0(M/M0)^(1/3) (see S(M) from Av 2014)
                        '''
                        Smain = self.S[-1][0]
                        self.S[-1][0] = Smain*((Mmain - Mfrag)/Mmain)**(2./3.)

                        ''' 
                            update the strength of the main body
                            post fragmentation
                        '''
                        sigmamain = self.sigma[-1][0]
                        alphamain = self.fraginfo["alpha"][0]
                        self.sigma[-1][0] = sigmamain*(Mmain/(Mmain-Mfrag))**(alphamain)

                        ''' fill in the values for the fragment '''
                        self.M[-1][i]     = Mfrag
                        self.S[-1][i]     = surface_area(rfrag)#Smain*(Mfrag/Mmain)**(2./3.)
                        self.r[-1][i]     = np.sqrt(self.S[-1][i]/(np.pi))
                        self.theta[-1][i] = self.theta[-1][0]
                        
                        if(verbose==2):
                            print("Frag at h=%.3f km, P=%.3f MPa, Mfrag=%.3e kg, Mmain: %.3e kg, rfrag=%.3f m"%(self.h[-1][0]/1000., self.DynPres[-1][0]/1.e6, Mfrag,  Mmain, rfrag))
                            print("Smain before: %.3f after: %.3f"%(Smain, self.S[-1][0]))

                        ''' 
                            set the strength of the fragment based on the same 
                            mass-strength scaling as the main body
                            unless it's specified separately
                        '''
                        if(self.fraginfo["sigmafrag"][i] == -1):
                            sigmafrag = self.sigma[-1][0]*(Mmain/Mfrag)**self.fraginfo["alpha"][0]
                        else:
                            sigmafrag = self.fraginfo["sigmafrag"][i]
                        self.fraginfo["sigmafrag"][i] = sigmafrag
                        self.sigma[-1][i] = sigmafrag

                        self.v[-1][i]     = self.v[-1][0]
                        self.h[-1][i]     = self.h[-1][0]
                        self.Mfr[-1][i]   = Mfrag
                        self.Nfr[-1][i]   = 1

            ### update the time 
            t  += dt
            self.t.append(t)

            ### stop conditions
            # if(self.v[-1][0] < vlim):
            #     end = True
            # if(self.M[-1][0] < mlim):
            #     end = True
            if(self.t[-1] > tlim):
                end = True
                for i in range(nfrags):
                    if(self.tend[i]==0.):
                        self.hend[i]  = h
                        self.tend[i]  = t
                        self.vend[i]  = v
                        self.mend[i]  = M

        self.t = np.asarray(self.t)
        self.M = np.asarray(self.M)
        self.v = np.asarray(self.v)
        self.r = np.asarray(self.r)
        self.S = np.asarray(self.S)
        self.h = np.asarray(self.h)
        self.theta = np.asarray(self.theta)
        self.DynPres = np.asarray(self.DynPres)
        self.sigma = np.asarray(self.sigma)
        self.Mfr = np.asarray(self.Mfr)
        self.Nfr = np.asarray(self.Nfr)
        self.dErdt = np.asarray(self.dErdt)
        self.dEddt = np.asarray(self.dEddt)


        # self.t = self.t.reshape((self.t.shape[0],1))

        ''' clean up the data '''
        self.dEdtall = np.sum(self.dErdt, axis=1)
        if(offset_time):
            dtoff        = self.t[self.dEdtall.argmax()]
            self.t      -= dtoff
            self.tend   -= dtoff
        self.Et      = interp1d(self.t, np.log10(self.dEdtall+1.e-25), kind='cubic', bounds_error=False, fill_value=-25)

        self.Edepo    = np.zeros_like(self.dEddt)
        self.Edepoall = np.zeros_like(self.dEdtall)
        for i in range(nfrags):
            maski    = self.dEddt[:,i] > 0.
            self.Edepo[maski,i]  = (self.dEddt[maski,i])/(self.v[maski,i]*np.sin(self.theta[maski,i]))
            ### interpolate each Edepo on to the grid of the main body
            if(i > 0):
                hi       = self.h[maski,i]
                Edepoi   = self.Edepo[maski,i]

                Eh = interp1d(hi, np.log10(Edepoi), kind='cubic', bounds_error=False, fill_value=-25.)
                self.Edepoall += 10.**(Eh(self.h[:,0]))
            else:
                self.Edepoall += self.Edepo[:,i]
        
        if(verbose>0):
            kt = 4.184e12 ## 1 kt TNT in J
            ## convert dE/dt to dE/dz in units of kT/km
            #self.Edepo = self.dEddt[:,0]/(self.v[:,0]*np.sin(self.theta[:,0]))*(1000./kt)

            ''' print out the output at the end ''' 
            endt    = self.tend.max()#self.t[-1]
            tii     = np.where(self.t == endt)[0]

            endm    = self.mend.sum()#self.M[-1][0]
            endv    = self.vend[0]#self.v[-1][0]
            endh    = self.hend.min()#self.h[-1][0]
            endNfr  = self.Nfr[tii,0]
            endMfr  = self.Mfr[tii,0]
            endP    = self.Pz(endh)

            ermax   = self.dEdtall.max()
            ermaxi  = self.dEdtall.argmax()
            hrmax   = self.h[ermaxi,0]/1000.
            edmax   = self.Edepoall.max()*1000./kt
            edmaxi  = self.Edepoall.argmax()
            hdmax   = self.h[edmaxi,0]/1000.

            ## get the total energy released in kt
            totenergy = np.trapz(np.sum(self.dErdt,axis=1), self.t[:])/kt

            print("Simulation ended at: ")
            print(" t=%.3fs with net mass=%.3e kg at v=%.3f km/s at h=%.3f km (P=%.2f mbar)"%(endt, endm, endv/1000., endh/1000., endP/100.))
            print("Total of %6d fragments with mass %.3f kg."%(endNfr, endMfr))
            print("Total energy released: %.3e kt"%(totenergy))
            print("Energy release peak of %.3e kt/s at h=%.3f km P=%.2f mbar"%(ermax/kt, hrmax, self.Pz(hrmax*1000.)/100.))
            print("Energy deposition peak of %.3e kt/km at h=%.3f km P=%.2f mbar"%(edmax, hdmax, self.Pz(hdmax*1000.)/100.))
            print()

