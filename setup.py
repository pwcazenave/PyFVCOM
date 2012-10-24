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
    name = "fvcom-py",
    version = "0.1",
    author = "Pierre Cazenave",
    author_email = "pwcazenave@gmail.com",
    description = ("fvcom-py is a collection of various tools and utilities which can be used to extract, analyse and plot input and output files from FVCOM."),
    license = "GPL",
    keywords = "fvcom",
    platforms = "any",
    url = "https://bitbucket.org/pwcazenave/fvcom-py",
    packages=['fvcom-py'],
    long_description=read('README.md'),
    classifiers=classifiers
)

