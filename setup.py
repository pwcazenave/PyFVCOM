import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

classifiers = """\
Development Status :: alpha
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: GPL
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
"""

setup(
    name = "PyFVCOM",
    version = "1.2",
    author = "Pierre Cazenave",
    author_email = "pica@pml.ac.uk",
    description = ("PyFVCOM is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM."),
    license = "MIT",
    keywords = "fvcom, unstructured grid, mesh",
    platforms = "any",
    url = "https://gitlab.ecosystem-modelling.pml.ac.uk/pica/PyFVCOM",
    packages=['PyFVCOM'],
    long_description=read('README.md'),
    classifiers=classifiers
)

