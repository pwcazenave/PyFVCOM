import numpy as np
import matplotlib.pyplot as plt

x = np.array([  1.00000000e-06,   1.00000000e-05,   1.00000000e-04,
         1.00000000e-03,   1.00000000e-02,   1.00000000e-01,
         1.00000000e+00,   1.00000000e+01,   1.00000000e+02])
y = np.array([  6.19684473e+01,   6.19684477e+02,   6.19684518e+03,
         6.19684930e+04,   6.19689049e+05,   6.19730251e+06,
         6.20143597e+07,   6.24437679e+08,   6.68337296e+09])


# Our model is y = a * x, so things are quite simple, in this case...
# x needs to be a column vector instead of a 1D vector for this, however.
x = x[:,np.newaxis]
a, _, _, _ = np.linalg.lstsq(x, y)

plt.plot(x, y, 'bo')
plt.plot(x, a*x, 'r-')
plt.show()
