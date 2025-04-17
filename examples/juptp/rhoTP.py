# Jarrad Pond

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('juptp/update_jov_810.txt',skiprows=1,)


grid = data[:,0]

rho = data[:,1]

T = data[:,2]

p = data[:,3]*1.e-3 ## dyne/cm^2 to mbar

z = data[:,4]/1.e5

data = np.zeros((z.shape[0],3))

data[:,0] = z
data[:,1] = p
data[:,2] = T

plt.plot(T, z)
plt.show()

np.savetxt("juptp/jupiter.tp", data, delimiter=",")