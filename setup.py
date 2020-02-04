from setuptools import setup

version = '2.2.0'

setup(
    name='PyFVCOM',
    packages=['PyFVCOM', 'PyFVCOM.ctd', 'PyFVCOM.grid', 'PyFVCOM.utilities'],
    version=version,
    description=("PyFVCOM is a collection of various tools which can be used to generate inputs files and interact "
                 "with FVCOM outputs."),
    author='Pierre Cazenave',
    author_email='pica@pml.ac.uk',
    url='https://gitlab.ecosystem-modelling.pml.ac.uk/fvcom/PyFVCOM',
    download_url='http://gitlab.em.pml.ac.uk/fvcom/PyFVCOM/repository/archive.tar.gz?ref={}'.format(version),
    keywords=['fvcom', 'unstructured grid', 'mesh'],
    license='MIT',
    platforms='any',
    install_requires=['jdcal', 'lxml', 'matplotlib', 'netCDF4', 'networkx', 'numpy>=1.13.0', 'pandas', 'xlsxwriter',
                      'pyproj', 'pytz', 'scipy', 'pyshp', 'UTide', 'shapely', 'descartes'],
    classifiers=[]
)
