#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# Fiddling around with matplotlib
plt.plot([0,1,2,3], [10,20,30,40])
plt.plot([0,1,2,3], [10,20,30,40], 'ro')
plt.ylabel('Some variables')
plt.xlabel('Some constants')
plt.axis('equal')
plt.axis('tight')

plt.show()

# Do some more interesting lines
t = np.arange(0., 5., 0.1)
plt.plot(t, t, 'gx')
plt.plot(t, t**2, 'x')
plt.plot(t, t**3, 'kx')
plt.ylabel('Some variables')
plt.xlabel('Some constants')
plt.axis('equal')
plt.axis('tight')

plt.show()

