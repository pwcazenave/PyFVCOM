
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt

from cmocean import cm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyFVCOM.read import ncread


# In[3]:

# Load the model output.
fvcom = 'sample.nc'

# Extract only the first 20 time steps.
dims = {'time': ':20'}

# And only these variables
varlist = ('lon', 'lat', 'nv', 'zeta', 'Times')

FVCOM = ncread(fvcom, vars=varlist, dims=dims, noisy=True)


# In[4]:

# Lay the groundwork for a plot of the model surface.

triangles = FVCOM['nv'].transpose() - 1  # offset for Python indexing.

extents = np.array((FVCOM['lon'].min(),
                    FVCOM['lon'].max(),
                    FVCOM['lat'].min(),
                    FVCOM['lat'].max()))

m = Basemap(llcrnrlon=extents[0],
            llcrnrlat=extents[2],
            urcrnrlon=extents[1],
            urcrnrlat=extents[3],
            rsphere=(6378137.00, 6356752.3142),
            resolution='h',
            projection='merc',
            lat_0=extents[-2:].mean(),
            lon_0=extents[:2].mean(),
            lat_ts=extents[:2].mean())

parallels = np.arange(np.floor(extents[2]), np.ceil(extents[3]), 1)
meridians = np.arange(np.floor(extents[0]), np.ceil(extents[1]), 2)

x, y = m(FVCOM['lon'], FVCOM['lat'])


# In[5]:

# Plot the surface elevation at the 6th time step.

fig = plt.figure(figsize=(10, 10))  # units are inches for the size
ax = fig.add_subplot(111)

tp = ax.tripcolor(x, y, triangles, FVCOM['zeta'][5, :], cmap=cm.balance)
tp.set_clim(-5, 5)  # clip the colours to +/- 5m.

# Add the coastline.
m.drawmapboundary()
m.drawcoastlines()
m.fillcontinents(color='0.6')
m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0)
m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0)

# # Add a nice colour bar.
# div = make_axes_locatable(ax)
# cax = div.append_axes("right", size="5%", pad=0.2)
# cb = fig.colorbar(tp, cax=cax)
# cb.set_label("Surface elevation (m)")

ax.set_title(''.join(FVCOM['Times'][-1, :-7].astype(str)))


# In[ ]:



