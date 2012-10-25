"""
Replicate the tidal ellipse example file from Zhigang Xu's tidal_ellipse MATLAB
toolbox.

Pierre Cazenave (Plymouth Marine Laboratory), October 2012.

"""

import matplotlib.pyplot as plt
import numpy as np

from tidal_ellipse import *

# Demonstrate how to use ap2ep and ep2ap
Au = np.random.random([4,3,2]);         # so 4x3x2 multi-dimensional matrices
Av = np.random.random([4,3,2]);         # are used for the demonstration.
Phi_v = np.random.random([4,3,2])*360;  # phase lags inputs are expected to
Phi_u = np.random.random([4,3,2])*360;  # be in degrees.

plt.figure(1)
plt.clf()
SEMA, ECC, INC, PHA, w = ap2ep(Au, Phi_u, Av, Phi_v, [2, 3, 1])
plt.figure(2)
plt.clf()
rAu, rPhi_u, rAv, rPhi_v, rw = ep2ap(SEMA, ECC, INC, PHA, [2, 3, 1])

# Check if ep2ap has recovered Au, Phi_u, Av, Phi_v
print np.max(np.abs(rAu-Au).flatten())        #  = 9.9920e-16, = 2.22044604925e-16
print np.max(np.abs(rAv-Av).flatten())        #  = 6.6613e-16, = 7.77156117238e-16
print np.max(np.abs(rPhi_u-Phi_u).flatten())  #  = 4.4764e-13, = 1.70530256582e-13
print np.max(np.abs(rPhi_v-Phi_v).flatten())  #  = 1.1369e-13, = 2.27373675443e-13
print np.max(np.max(np.abs(w-rw).flatten()))  #  = 1.3710e-15, = 1.1322097734e-15
# For the random realization I (Zhigang Xu) had, the differences are listed on
# the right hand of the above column. I (Pierre Cazenave) got the second column
# with the Python version. What are yours?

# Zhigang Xu
# Nov. 12, 2000
# Pierre Cazenave
# October, 2012

