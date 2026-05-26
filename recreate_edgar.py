import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ABS_QE_Calibration.txt')

print(data)
plt.plot(data[:,0], data[:,1], 'o')
plt.plot(data[:,0], data[:,2], 'o')
plt.plot(data[:,0], data[:,3], 'o')
plt.plot(data[:,0], data[:,4], 'o')
plt.yscale('log')
plt.show()

