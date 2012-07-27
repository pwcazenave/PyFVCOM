#!/usr/bin/env python

# Create a 2D netCDF file.
from Scientific.IO.NetCDF import NetCDFFile as Dataset
#from netCDF4_classic import Dataset
from numpy import arange, dtype # array module from http://numpy.scipy.org

# the output array to write will be nx x ny
nx = 6; ny = 12
# open a new netCDF file for writing.
ncfile = Dataset('simple_xy.nc','w') 
# create the output data.
data_out = arange(nx*ny) # 1d array
data_out.shape = (nx,ny) # reshape to 2d array
# create the x and y dimensions.
ncfile.createDimension('x',nx)
ncfile.createDimension('y',ny)
# create the variable (4 byte integer in this case)
# first argument is name of variable, second is datatype, third is
# a tuple with the names of dimensions.
data = ncfile.createVariable('data',dtype('int32').char,('x','y'))
# write data to variable.
data[:] = data_out
# close the file.
ncfile.close()
print '*** SUCCESS writing example file simple_xy.nc!'

