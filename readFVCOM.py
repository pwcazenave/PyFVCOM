from netCDF4 import Dataset, MFDataset

def readFVCOM(file, varList, noisy=False):
    """
    Read in the FVCOM results file and spit out numpy arrays for
    each of the variables.
    """

    rootgrp = Dataset(file, 'r')
    mfdata = MFDataset(file)

    if noisy:
        print "File format: " + rootgrp.file_format

    FVCOM = {}
    for key, var in rootgrp.variables.items():
        if noisy:
            print 'Found ' + key,

        if key in varList:
            if noisy:
                print '(extracted)'
            FVCOM[key] = mfdata.variables[key][:]
        else:
            if noisy:
                print

    return FVCOM
