""" 
Edits the spectide amplitude values to some factor of their original value.

WARNING: When using this on FVCOM input files, it will change the format of
the variables. changeNC presumes each variable has a value and unit associated
with it, whereas some of the variables in the FVCOM inputs are in fact not 
that sort of data, and so have different structures. Probably best to use the
combination of ncdump and ncgen to edit the values as text and generate a new
NetCDF file from that edited text.

"""

from changeNC import *

infile = './co2_spectide.nc'
outfile = './co2_spectide_scaled.nc'
scaleFact = 0.75

av = AutoVivification()

av['tide_Eamp']['convert'] = lambda x:x*scaleFact

changeNC(infile, outfile, av)
