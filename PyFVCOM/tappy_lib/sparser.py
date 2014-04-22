#!/usr/bin/env python

"""
NAME:
    sparser.py

SYNOPSIS:
    sparser.py [options] filename

DESCRIPTION:
    The sparser.py script is a Specified PARSER.  It is unique (as far as I can
    tell) because it doesn't care about the delimiter(s).  The user specifies
    what is expected, and the order, for each line of text.  All of the heavy
    lifting is handled by pyparsing (http://pyparsing.sf.net).

OPTIONS:
    -h,--help        this message
    -v,--version     version
    -d,--debug       turn on debug messages

EXAMPLES:
    1. As standalone
        sparser.py myfile
    2. As library
        import sparser
        ...

#Copyright (C) 2006  Tim Cera timcera@earthlink.net
#
#
#    This program is free software; you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by the Free
#    Software Foundation; either version 2 of the License, or (at your option)
#    any later version.
#
#    This program is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#    or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#    for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    675 Mass Ave, Cambridge, MA 02139, USA.
"""

#===imports======================
import sys
import os
import getopt
import re
import gzip
import datetime

#from tappy_lib.pyparsing.pyparsing import *
from pyparsing import *


#===globals======================
modname = "sparser"
__version__ = "0.3"


#--option args--
debug_p = 0
#opt_b=None  #string arg, default is undefined


#---positional args, default is empty---
pargs = []


#---other---


#===utilities====================
def msg(txt):
    """Send message to stdout."""
    sys.stdout.write(txt)
    sys.stdout.flush()

def debug(ftn, txt):
    """Used for debugging."""
    if debug_p:
        sys.stdout.write("%s.%s:%s\n" % (modname, ftn, txt))
        sys.stdout.flush()

def fatal(ftn, txt):
    """If can't continue."""
    msg = "%s.%s:FATAL:%s\n" % (modname, ftn, txt)
    raise SystemExit, msg

def usage():
    """Prints the docstring."""
    print __doc__


# Will hold list of pyparsing constructs.
grammar = []

# Set a default value for decimal_separator.
decimal_sep = "."

# Extra dictionary
extra_dict = {}

#====================================
class DefinitionFileNotFoundError(Exception):
    def __init__(self, def_filename):
        self.value = def_filename
    def __str__(self):
        return repr(self.value)

def toInteger(instring, loc, tokenlist):
    """Converts parsed integer string to an integer."""
    return int(tokenlist[0])

def toFloat(instring, loc, tokenlist):
    """Converts parsed real string to a real."""
    return float(tokenlist[0])

def toString(instring, loc, tokenlist):
    """Returns an integer or real as a string."""
    return tokenlist[0]

def toDatetime(instring, loc, tokenlist):
    """Returns a datetime object."""
    global _origin
    global _unit

    exec('rvar = _origin + datetime.timedelta({0}={1})'.format(_unit, float(tokenlist[0])))
    return rvar

def isotoDate(instring, loc, tokenlist):
    """Returns a datetime object."""
    rvar = [int(i) for i in tokenlist[::2]]
    rvar = datetime.datetime(*rvar)
    return rvar

def integer(name,
            minimum=1,
            maximum=None,
            exact=None,
            sign=Optional(oneOf("- +")),
            parseAct=toInteger):
    """Appends a skip/integer combination to the parse constructs."""
    lint = Combine(sign +
                   Word(nums,
                        min=minimum,
                        max=maximum,
                        exact=exact))
    grammar.append(SkipTo(lint))
    grammar.append(lint
                   .setResultsName(name)
                   .setParseAction(parseAct))

def positive_integer(name,
                     minimum=1,
                     maximum=None,
                     exact=None):
    """Will only parse a positive integer."""
    integer(name,
            minimum=minimum,
            maximum=maximum,
            exact=exact,
            sign=Optional("+"))

def negative_integer(name,
                     minimum=1,
                     maximum=None,
                     exact=None):
    """Will only parse a negative integer."""
    integer(name,
            minimum=minimum,
            maximum=maximum,
            exact=exact,
            sign="-")

def real(name,
         required_decimal=True,
         sign=Optional(oneOf("- +")),
         parseAct=toFloat):
    """Appends a skip/real pair to the parse constructs."""
    if required_decimal:
        lword = Combine(sign +
                    Regex('[0-9]*\.[0-9]*') +
                    Optional(oneOf("E e D d") + Optional(oneOf("- +")) + Word(nums)))
    else:
        lword = Combine(sign +
                    Word(nums + decimal_sep) +
                    Optional(oneOf("E e D d") + Optional(oneOf("- +")) + Word(nums)))
    grammar.append(SkipTo(lword))
    grammar.append(lword
                   .setResultsName(name)
                   .setParseAction(parseAct))

def positive_real(name,
                  minimum=1,
                  maximum=None,
                  exact=None):
    """Will only parse a positive real."""
    real(name,
         minimum=minimum,
         maximum=maximum,
         exact=exact,
         sign=Optional("+"))

def negative_real(name,
                  minimum=1,
                  maximum=None,
                  exact=None):
    """Will only parse a negative real."""
    real(name,
         minimum=minimum,
         maximum=maximum,
         exact=exact,
         sign="-")

def real_as_string(name,
                   minimum=1,
                   maximum=None,
                   exact=None,
                   sign=Optional(oneOf("- +")),
                   parseAct=toString):
    """Parses a real number, but returns it as a string."""
    real(name,
         minimum=minimum,
         maximum=maximum,
         exact=exact,
         sign=Optional("- +"),
         parseAct=parseAct)

def integer_as_string(name,
                      minimum=1,
                      maximum=None,
                      exact=None,
                      sign=Optional(oneOf("- +")),
                      parseAct=toString):
    """Parses an integer, but returns it as a string."""
    integer(name,
            minimum=minimum,
            maximum=maximum,
            exact=exact,
            sign=Optional("+"),
            parseAct=parseAct)

def isoformat_as_datetime(name,
         parseAct=isotoDate):
    """Appends a skip/real pair to the parse constructs."""
    lword = (Word(nums) + '-' +
             Word(nums) + '-' +
             Word(nums) + oneOf('T  ') +
             Word(nums) + ':' +
             Word(nums) + ':' +
             Word(nums)
             )
    grammar.append(SkipTo(lword))
    grammar.append(lword
                   .setResultsName(name)
                   .setParseAction(parseAct))

def real_as_datetime(name,
                     sign=Optional(oneOf("- +")),
                     origin=datetime.datetime(1900,1,1),
                     unit='days',
                     parseAct=toDatetime):
    global _origin
    global _unit
    _origin = origin
    _unit = unit
    real(name,
         sign=Optional("- +"),
         parseAct=toDatetime)

def integer_as_datetime(name,
                      minimum=1,
                      maximum=None,
                      exact=None,
                      sign=Optional(oneOf("- +")),
                      origin=datetime.datetime(1900,1,1),
                      unit='days',
                      parseAct=toDatetime):
    global _origin
    global _unit
    _origin = origin
    _unit = unit
    integer(name,
         minimum=minimum,
         maximum=maximum,
         exact=exact,
         sign=Optional("- +"),
         parseAct=toDatetime)

def qstring(name):
    """Parses a quoted (either double or single quotes) string."""
    quoted_string = sglQuotedString | dblQuotedString
    grammar.append(SkipTo(quoted_string))
    grammar.append(quoted_string.setResultsName(name))

def delimited_as_string(name):
    """Parses out any delimited group as a string."""
    wrd = Word(alphanums)
    grammar.append(SkipTo(wrd))
    grammar.append(wrd.setResultsName(name))

def number_as_real(name,
                    sign=Optional(oneOf("- +")),
                    parseAct=toFloat):
    """Parses any number as a real."""
    real(name,
         required_decimal=False,
         sign=Optional(oneOf("- +")),
         parseAct=toFloat)

def number_as_integer(name,
                    minimum=1,
                    maximum=None,
                    exact=None,
                    sign=Optional(oneOf("- +")),
                    parseAct=toInteger):
    """Parses any number as a integer."""
    integer(name,
         minimum=1,
         maximum=None,
         exact=None,
         sign=Optional(oneOf("- +")),
         parseAct=toInteger)

def number_as_string(name,
                    minimum=1,
                    maximum=None,
                    exact=None,
                    sign=Optional(oneOf("- +")),
                    parseAct=toString):
    """Parses any number as a string."""
    real(name,
         minimum=1,
         maximum=None,
         exact=None,
         sign=Optional(oneOf("- +")),
         parseAct=toString)

def insert(name, value):
    extra_dict[name] = value


class ParsedString(str):
    """
    String class inherited from 'str' plus a dictionary of parsed
    values and a line number.
    """
    def __init__(self, *args, **kw):
        str.__init__(args, kw)
        self.parsed_dict = {}
        self.line_number = 0


class ParseFileLineByLine:
    """
    Bring data from text files into a program, optionally parsing each line
    according to specifications in a parse definition file.

    ParseFileLineByLine instances can be used like normal file objects (i.e. by
    calling readline(), readlines(), and write()), but can also be used as
    sequences of lines in for-loops.

    If the 'filelike' module is available ParseFileLineByLine objects also
    handles compression transparently. i.e. it is possible to read lines from a
    compressed text file as if it were not compressed.  Compression is deduced
    from the file name suffixes '.Z' (compress/uncompress), '.gz'
    (gzip/gunzip), and '.bz2' (bzip2).  With 'filelike' you can also pass a URL
    as filename and ParseFileLineByLine will download data from the URL.

    The parse definition file name is developed based on the input file name.
    If the input file name is 'basename.ext', then the definition file is
    'basename.def'.  If a definition file specific to the input file is not
    found, then the program searches for the file 'sparse.def' which would be
    the definition file for all files in that directory without a file specific
    definition file.

    Constructor:
    ParseFileLineByLine(|filename|, |mode|='"r"'), where |filename| is the name
    of the file (compressed or a URL if 'filelike' module is available) and
    |mode| is one of '"r"' (read), '"w"' (write) or '"a"' (append, not
    supported for .Z files).
    """

    def __init__(self, filename, def_filename=None, mode = 'r'):
        """Opens input file, and if available the definition file.  If the
        definition file is available __init__ will then create some pyparsing
        helper variables.  """

        filen, file_extension = os.path.splitext(filename)

        # I use filelike which allows you to open up compressed files and urls
        # as files.  Test to see whether it is available.
        try:
            import filelike
            tmp_open = filelike.open
        except ImportError:
            tmp_open = open
        self.file = tmp_open(filename, mode)

        # Try to maintain a line count
        self.record_number = 0

        self.grammar = None
        self.parsedef = None

        # Try to find a parse ('*.def') definition file.  First try to find
        # a file specific parse definition file, then look for 'sparse.def'
        # that would be the definition file for all files within the directory.

        # The definition file is pure Python.  The one variable that needs to
        # be specified is 'parse'.  The 'parse' variable is a list of functions
        # defining the name, type, and because it is a list, the order of
        # variables on each line in the data file.  The variable name is a
        # string, the type variable is defined as integer, real, and qString.

        # parse = [
        #          integer('state', exact=3),
        #          integer('station', exact=4),
        #          positive_integer('month'),
        #          integer('day'),
        #          integer('year'),
        #          integer('hour'),
        #          integer('minute'),
        #          positive_real('water_level'),
        #         ]

        self.line_number = 0

        definition_file_one = filen + ".def"
        if os.path.dirname(filen):
            definition_file_two = os.path.dirname(filen) + os.sep + "sparse.def"
        else:
            definition_file_two = "sparse.def"

        if os.path.exists(definition_file_two):
            self.parsedef = definition_file_two
        if os.path.exists(definition_file_one):
            self.parsedef = definition_file_one
        if def_filename:
            if os.path.exists(def_filename):
                self.parsedef = def_filename
            else:
                raise DefinitionFileNotFoundError(def_filename)
        if self.parsedef:
            execfile(self.parsedef)
            self.grammar = And(grammar[1:] + [restOfLine])

    def __del__(self):
        """Delete (close) the file wrapper."""
        self.close()

    def __getitem__(self, item):
        """Used in 'for line in fp:' idiom."""
        line = self.readline()
        if not line:
            raise IndexError
        return line

    def readline(self):
        """Reads (and optionally parses) a single line."""
        line = self.file.readline()
        line = ParsedString(line)
        self.line_number = self.line_number + 1
        line.line_number = self.line_number
        if self.grammar and line:
            try:
                line.parsed_dict = self.grammar.parseString(line).asDict()
                for key in extra_dict.keys():
                    line.parsed_dict[key] = extra_dict[key]
            except ParseException:
                line.parsed_dict = {}
        return line

    def readlines(self):
        """Returns a list of all lines (optionally parsed) in the file."""
        if self.grammar:
            tot = []
            # Used this way instead of a 'for' loop against
            # self.file.readlines() so that there wasn't two copies of the file
            # in memory.
            while 1:
                line = self.readline()
                if not line:
                    break
                tot.append(line)
            return tot
        return self.file.readlines()

    def write(self, data):
        """Write to a file."""
        self.file.write(data)

    def writelines(self, list):
        """Write a list to a file. Each item in the list is a line in the
        file.
        """
        for line in list:
            self.file.write(line)

    def close(self):
        """Close the file."""
        self.file.close()

    def flush(self):
        """Flush in memory contents to file."""
        self.file.flush()


#=============================
def main(pargs):
    """This should only be used for testing. The primary mode of operation is
    as an imported library.
    """
    input_file = sys.argv[1]
    fp = ParseFileLineByLine(input_file)
    for i in fp:
        print i


#-------------------------
if __name__ == '__main__':
    ftn = "main"
    opts, pargs = getopt.getopt(sys.argv[1:], 'hvd',
                 ['help', 'version', 'debug', 'bb='])
    for opt in opts:
        if opt[0] == '-h' or opt[0] == '--help':
            print modname+": version="+__version__
            usage()
            sys.exit(0)
        elif opt[0] == '-v' or opt[0] == '--version':
            print modname+": version="+__version__
            sys.exit(0)
        elif opt[0] == '-d' or opt[0] == '--debug':
            debug_p = 1
        elif opt[0] == '--bb':
            opt_b = opt[1]

    #---make the object and run it---
    main(pargs)

#===Revision Log===
#Created by mkpythonproj:
#2006-02-06  Tim Cera
#
