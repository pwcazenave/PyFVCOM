
import unittest
from StringIO import StringIO
import tempfile
import os

import filelike
from filelike import to_filelike, is_filelike, join, wrappers

class ProxyObject:
    def __init__(self,obj):
        self._obj = obj
    def __getattr__(self,attr):
        return getattr(self._obj,attr)


class Test_Read(unittest.TestCase):
    """Generic file-like testcases for readable files."""

    contents = "Once upon a time, in a galaxy far away,\nGuido van Rossum was a space alien."
    empty_contents = ""

    def makeFile(self,contents,mode):
        """This method must create a file of the type to be tested.

        The returned file must have the given contents and mode, and
        have a special method "getvalue" which will return the contents
        of the file as a string.

        By default, a temporary file object is used.  If our tests don't
        pass for the built-in file type, how can we expect to achieve
        anything with them?
        """
        mode = filter(lambda c: c in "rwa+t",mode)
        if "b" not in mode:
            mode = mode + "b"
        if "r" not in mode and "+" not in mode:
            mode = mode + "+"
        (fd,nm) = tempfile.mkstemp()
        os.write(fd,contents)
        os.close(fd)
        f = ProxyObject(open(nm,mode))
        def getvalue():
           pos = f.tell()
           f.seek(0)
           val = f.read()
           f.seek(pos)
           return val
        f.getvalue = getvalue
        return f

    def setUp(self):
        self.file = self.makeFile(self.contents,"r")

    def tearDown(self):
        self.file.close()

    def test_read_all(self):
        c = self.file.read()
        self.assertEquals(c,self.contents)

    def test_read_stream(self):
        f = self.makeFile(self.contents,"r-")
        self.assertEquals(f.read(),self.contents)

    def test_read_size(self):
        c = self.file.read(5)
        self.assertEquals(c,self.contents[:5])
        c = self.file.read(7)
        self.assertEquals(c,self.contents[5:12])

    def test_readline(self):
        c = self.file.readline()
        if self.contents.find("\n") < 0:
            extra = ""
        else:
            extra = "\n"
        self.assertEquals(c,self.contents.split("\n")[0]+extra)

    def test_readlines(self):
        cs = [ln.strip("\n") for ln in self.file.readlines()]
        self.assertEquals(cs,self.contents.split("\n"))

    def test_xreadlines(self):
        cs = [ln.strip("\n") for ln in self.file.xreadlines()]
        self.assertEquals(cs,self.contents.split("\n"))

    def test_read_empty_file(self):
        f = self.makeFile(self.empty_contents,"r")
        self.assertEquals(f.read(),self.empty_contents)

    def test_eof(self):
        self.file.read()
        self.assertEquals(self.file.read(),"")
        self.assertEquals(self.file.read(),"")


class Test_ReadWrite(Test_Read):
    """Generic file-like testcases for writable files."""

    def setUp(self):
        self.file = self.makeFile(self.contents,"r+")

    def test_write(self):
        f = self.makeFile(self.empty_contents,"w")
        f.write(self.contents)
        self.assertEquals(f.tell(),len(self.contents))
        f.flush()
        self.assertEquals(f.getvalue(),self.contents)
        f.close()

    def test_append(self):
        f = self.makeFile(self.empty_contents,"a")
        f.write(self.contents)
        self.assertEquals(f.tell(),len(self.contents))
        f.flush()
        self.assertEquals(f.getvalue(),self.contents)
        f.close()

    def test_write_stream(self):
        f = self.makeFile(self.empty_contents,"w-")
        f.write(self.contents)
        self.assertEquals(f.tell(),len(self.contents))
        f.flush()
        self.assertEquals(f.getvalue(),self.contents)
        f.close()

    def test_write_read(self):
        self.file.write("hello")
        self.file.seek(0,1)
        c = self.file.read()
        self.assertEquals(c,self.contents[5:])

    def test_read_write_read(self):
        c = self.file.read(5)
        self.assertEquals(c,self.contents[:5])
        self.file.write("hello")
        c = self.file.read(5)
        self.assertEquals(c,self.contents[10:15])


class Test_ReadWriteSeek(Test_ReadWrite):
    """Generic file-like testcases for seekable files."""

    def test_seek_tell(self):
        self.assertEquals(self.file.tell(),0)
        self.file.seek(7)
        self.assertEquals(self.file.tell(),7)
        self.assertEquals(self.file.read(),self.contents[7:])
        self.file.seek(0,0)
        self.assertEquals(self.file.tell(),0)

    def test_read_write_seek(self):
        c = self.file.read(5)
        self.assertEquals(c,self.contents[:5])
        self.file.seek(0,1)
        self.file.write("hello")
        self.assertEquals(self.file.tell(),10)
        self.file.seek(0)
        self.assertEquals(self.file.tell(),0)
        c = self.file.read(10)
        self.assertEquals(c,self.contents[:5] + "hello")

    def test_seek_cur(self):
        self.assertEquals(self.file.tell(),0)
        self.file.seek(7,1)
        self.assertEquals(self.file.tell(),7)
        self.file.seek(7,1)
        self.assertEquals(self.file.tell(),14)
        self.file.seek(-5,1)
        self.assertEquals(self.file.tell(),9)

    def test_seek_end(self):
        self.assertEquals(self.file.tell(),0)
        self.file.seek(-7,2)
        self.assertEquals(self.file.tell(),len(self.contents)-7)
        self.file.seek(3,1)
        self.assertEquals(self.file.tell(),len(self.contents)-4)

    def test_write_at_end(self):
        self.assertEquals(self.file.tell(),0)
        self.file.seek(0,2)
        self.file.write("testable")
        self.file.seek(0,0)
        self.assertEquals(self.file.read(),self.contents+"testable")

    def test_write_twice(self):
        f = self.makeFile(self.empty_contents,"w")
        f.write(self.contents)
        self.assertEquals(f.tell(),len(self.contents))
        f.flush()
        self.assertEquals(f.getvalue(),self.contents)
        f.seek(-5,2)
        self.assertEquals(f.tell(),len(self.contents) - 5)
        f.write(self.contents[-5:])
        f.flush()
        self.assertEquals(f.getvalue(),self.contents)


class Test_StringIO(Test_ReadWriteSeek):
    """Run our testcases against StringIO, basically to test the tests."""

    def makeFile(self,contents,mode):
        f = StringIO(contents)
        f.seek(0)
        def xreadlines():
            for ln in f.readlines():
                yield ln
        f.xreadlines = xreadlines
        return f


class Test_Join(Test_ReadWriteSeek):
    """Run our testcases against filelike.join."""

    def makeFile(self,contents,mode):
        files = []
        files.append(StringIO(contents[0:5]))
        files.append(StringIO(contents[5:8]))
        files.append(StringIO(contents[8:]))
        f = join(files)
        def getvalue():
            return "".join([s.getvalue() for s in files])
        f.getvalue = getvalue
        return f


class Test_IsTo(unittest.TestCase):
    """Tests for is_filelike/to_filelike."""

    def test_isfilelike(self):
        """Test behaviour of is_filelike."""
        self.assert_(is_filelike(tempfile.TemporaryFile()))
        self.assert_(is_filelike(tempfile.TemporaryFile("r"),"r"))
        self.assert_(is_filelike(tempfile.TemporaryFile("r"),"w"))
        self.assert_(is_filelike(StringIO()))

    def test_tofilelike_read(self):
        """Test behavior of to_filelike for mode "r-"."""
        class F:
            def read(self,sz=-1):
                return ""
        f = to_filelike(F(),"r-")
        self.assertEquals(f.__class__,wrappers.FileWrapper)
        self.assertEquals(f.read(),"")
        self.assertRaises(ValueError,to_filelike,F(),"r")
        self.assertRaises(ValueError,to_filelike,F(),"w-")
        self.assertRaises(ValueError,to_filelike,F(),"rw")

    def test_tofilelike_readseek(self):
        """Test behavior of to_filelike for mode "r"."""
        class F:
            def read(self,sz=-1):
                return ""
            def seek(self,offset,whence):
                pass
        f = to_filelike(F(),"r")
        self.assertEquals(f.__class__,wrappers.FileWrapper)
        self.assertEquals(f.read(),"")
        self.assertRaises(ValueError,to_filelike,F(),"w")
        self.assertRaises(ValueError,to_filelike,F(),"w-")
        self.assertRaises(ValueError,to_filelike,F(),"rw")

    def test_tofilelike_write(self):
        """Test behavior of to_filelike for mode "w-"."""
        class F:
            def write(self,data):
                pass
        f = to_filelike(F(),"w-")
        self.assertEquals(f.__class__,wrappers.FileWrapper)
        self.assertRaises(ValueError,to_filelike,F(),"w")
        self.assertRaises(ValueError,to_filelike,F(),"r")
        self.assertRaises(ValueError,to_filelike,F(),"r-")
        self.assertRaises(ValueError,to_filelike,F(),"rw")

    def test_tofilelike_writeseek(self):
        """Test behavior of to_filelike for mode "w"."""
        class F:
            def write(self,data):
                pass
            def seek(self,offset,whence):
                pass
        f = to_filelike(F(),"w")
        self.assertEquals(f.__class__,wrappers.FileWrapper)
        self.assertRaises(ValueError,to_filelike,F(),"r")
        self.assertRaises(ValueError,to_filelike,F(),"r-")

    def test_tofilelike_readwrite(self):
        """Test behavior of to_filelike for mode "rw"."""
        class F:
            def write(self,data):
                pass
            def read(self,sz=-1):
                return ""
            def seek(self,offset,whence):
                pass
        f = to_filelike(F(),"rw")
        self.assertEquals(f.__class__,wrappers.FileWrapper)
        self.assertEquals(f.read(),"")

    def test_tofilelike_stringio(self):
        """Test behaviour of to_filelike on StringIO instances."""
        f = to_filelike(StringIO())
        self.assert_(isinstance(f,StringIO))

    def test_tofilelike_string(self):
        """Test behaviour of to_filelike on strings."""
        f = to_filelike("testing")
        self.assert_(isinstance(f,StringIO))
        self.assertEquals(f.read(),"testing")
        

class Test_Docs(unittest.TestCase):
    """Unittests for our documentation."""

    def test_README(self):
        """Ensure that the README is in sync with the docstring.

        This test should always pass; if the README is out of sync it just
        updates it with the contents of filelike.__doc__.
        """
        dirname = os.path.dirname
        readme = os.path.join(dirname(dirname(__file__)),"README.txt")
        if not os.path.isfile(readme):
            f = open(readme,"wb")
            f.write(filelike.__doc__.encode())
            f.close()
        else:
            f = open(readme,"rb")
            if f.read() != filelike.__doc__:
                f.close()
                f = open(readme,"wb")
                f.write(filelike.__doc__.encode())
                f.close()



def build_test_suite():
    suite = unittest.TestSuite()
    for (dirnm,_,files) in os.walk(os.path.dirname(filelike.__file__)):
        for fname in files:
            if fname.startswith("test") and fname.endswith(".py"):
                testname = fname[:-3]
                modnames = []
                remainder = dirnm
                while "filelike" not in  modnames:
                    (remainder,modparent) = os.path.split(remainder)
                    modnames.insert(0,modparent)
                testmod = __import__(".".join(modnames),fromlist=(testname,))
                testmod = getattr(testmod,testname)
                for nm in dir(testmod):
                    if nm.startswith("Test"):
                        suite.addTest(unittest.makeSuite(getattr(testmod,nm)))
    return suite
