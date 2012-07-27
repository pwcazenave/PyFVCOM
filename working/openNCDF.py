#! /usr/bin/env matplotlib 
"""opens netCDF files in python, includes functions:
      listNCDF(netcdfObject): lists the contents of the netcdf-file
      varInfo('VARIABLE'): gives details of variable
   for extracting array sections use 
      Var=nCDF.var('VARIABLE')[t1:t2,z1:z2,i1:i2,j1:j2].squeeze()
   end plot with 
    1D:  pylab.plot(Var)
         pylab.show()
   or
    hovmoeller: pylab.pcolor(Var)
                pylab.show()
   see also matplotlib web page..."""
from pycdf import *
import os
import sys
import numpy
from pylab import *
from matplotlib import *

if len(sys.argv) < 2:
    print str( [el for el in os.listdir(os.getcwd()) if el[-3:] =='.nc'])
    filename=input('Give netCDF file name: ')
else:
    filename=sys.argv[1]

try: #check if filename is a ncdffile
    nCDF=CDF(filename,NC.NOWRITE)
    nCDF.automode()
    print 'nCDF-file %s opened...' % filename
    print 'created ncdf Object nCDF'
except CDFError,err:
    print "pycdf reported an error in function/method:",err[0]
    print "      netCDF error ",err[1],":",err[2]
    sys.exit()

varList=nCDF.variables().items()
varList.sort(cmp=lambda x,y: x[1][3]-y[1][3])

def listNCDF(ncdf):
        infoStr='-----------------\n'+\
            'nCDF Object:\n'+\
            '-----------------'

        for key,value in ncdf.attributes().items():
            infoStr+='\n\n'+key+':\t'+str(value)

        infoStr+='\n\n'+'Dimensions:\t(length,position,unlimited)\n'
        dimList=ncdf.dimensions(full=1).items()
        dimList.sort(cmp=lambda x,y: x[1][1]-y[1][1])

        for key,value in dimList:
            infoStr+='\n\t'+key
            if value[2]: 
                infoStr+='\tUNLIMITED => '+str(value[0])
            else:
                infoStr+='\t'+str(value[0])

        infoStr+='\n\n'+'Variables:\t(dimensions:size,position,type\n\t(1=BYTE,2=CHAR,3=SHORT,4=INT,5=FLOAT,6=DOUBLE)\n'
        #varList=ncdf.variables().items()
        #varList.sort(cmp=lambda x,y: x[1][3]-y[1][3])
        global varList

        for key,value in varList:
            infoStr+='\n\t'+key+':\t'+str(value[0])+'='+str(value[1])
            for k,v in ncdf.var(key).attributes().items():
                if k=='long_name':
                   infoStr+='\n\t\t'+str(v)
        return infoStr

def varInfo(varstr):

    global varList,nCDF

    for key,value in varList:
        if key==varstr:
            infoStr='\n\t'+key+':\t'+str(value[0])+': '+str(value[1])\
                    +'\t'+str(value[3])+'\t'+str(value[2])
            for k,v in nCDF.var(key).attributes().items():
                infoStr+='\n\t\t'+k+':\t'+str(v)+'\n'
            print infoStr
            return 
    print 'Variable "' + varstr + '" not found!'

    return

v=lambda var:numpy.ma.masked_where(
	nCDF.var(var)[:].squeeze()>10.**35,
	nCDF.var(var)[:].squeeze())
vpure=lambda var: nCDF.var(var)[:].squeeze()
listing=listNCDF(nCDF)

print listing
