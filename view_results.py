""" Quick script to take a look at a model output """


from readFVCOM import readFVCOM
from read_fvcom_results import animateModelOutput as ani

getVars = ['x', 'y', 'xc', 'yc', 'zeta', 'art1', 'h', 'time', 'TCO2', 'PH', 'DYE', 'siglev', 'salinity']
varOfInterest = 'zeta'

if varOfInterest not in getVars:
    print 'Warning: missing variable of interest in variables to be extracted'
    raise

base = '/data/medusa/pica/models/FVCOM/runCO2_leak'
in1 = base + '/output/scenarios/co2_S7_low_rate_full_tide_fvcom_0001.nc'
in2 = base + '/input/configs/inputV7/co2_grd.dat'

noisy = False

FVCOM = readFVCOM(in1, getVars, noisy)
ani(FVCOM, varOfInterest, 1, 1, 0, in2)
