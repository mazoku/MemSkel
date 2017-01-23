__author__ = 'Ryba'

from distutils.core import setup
import py2exe, sys, os
import matplotlib
import glob
#from glob import glob

def find_data_files(source,target,patterns):
    """Locates the specified data-files and returns the matches
    in a data_files compatible format.

    source is the root of the source data tree.
        Use '' or '.' for current directory.
    target is the root of the target data tree.
        Use '' or '.' for the distribution directory.
    patterns is a sequence of glob-patterns for the
        files you want to copy.
    """
    if glob.has_magic(source) or glob.has_magic(target):
        raise ValueError("Magic not allowed in src, target")
    ret = {}
    for pattern in patterns:
        pattern = os.path.join(source,pattern)
        for filename in glob.glob(pattern):
            if os.path.isfile(filename):
                targetpath = os.path.join(target,os.path.relpath(filename,source))
                path = os.path.dirname(targetpath)
                ret.setdefault(path,[]).append(filename)
    return sorted(ret.items())

def copy_all_files(source, target):
	if glob.has_magic(source) or glob.has_magic(target):
		raise ValueError("Magic not allowed in src, target")
	ret = {}
	for root, _, files in os.walk(source):
		for name in files:
			filename = os.path.join(root,name)
			targetpath = os.path.join(target,os.path.relpath(filename,source))
			path = os.path.dirname(targetpath)
			ret.setdefault(path,[]).append(filename)
	return sorted(ret.items())
	
file_data = copy_all_files('data', 'data')
file_libtiff = copy_all_files('c:\Python27\Lib\site-packages\libtiff', 'libtiff' )
	
sys.path.append('c:/Python27/Lib/site-packages/numpy/core/')

if os.path.isdir('c:/Program Files (x86)'):
	sys.path.append('c:\\Program Files (x86)\\Microsoft Visual Studio 9.0\\VC\\redist\\x86\\Microsoft.VC90.CRT')
else:
	sys.path.append('c:\\Program Files\\Microsoft Visual Studio 9.0\\VC\\redist\\x86\\Microsoft.VC90.CRT')

includes = ['numpy', 'matplotlib', 'matplotlib.backends.backend_qt4agg', 'Tkinter', 'ctypes', 'logging']
excludes = ['libtiff', '_ssl', 'doctest', 'pdb', 'email', 'pytz',
            '_gtkagg', '_tkagg', '_agg2', '_cairo', '_cocoaagg', '_fltkagg', '_gtk', '_gtkcairo', 'tc']

dist_dir = 'c:/Temp/MemSkel/'

data_files = [
            ('Microsoft.VC90.CRT', glob.glob(r'c:\Program Files\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\*.*'))
            ]
data_files.extend( matplotlib.get_py2exe_datafiles() )
data_files.extend( file_data )
data_files.extend( file_libtiff )
#data_files.extend( ('data', ['c:/Dropbox/Work/Python/MemSkel/data/*.*']) )
#data_files.extend( ('libtiff', glob(r'c:\Python27\Lib\site-packages\libtiff\*.*')) )
#data_files.extend( [os.path.join(libtiff_dir, 'libtiff.dll')] )
#dll_excludes = ['libtiff.dll']

setup(
    name='MemSkel',
    version='1.0',
    description='Software for segmenting membrane of cell and approximating it with B-spline. Works also for multi-page images' +
                ' in TIFF format. In that case, only segmentation of the first page is necesary. Resulting pages are segmented' +
                ' automatically. Corrections of results are possible.',
    author='Tomas Ryba',
    author_email='tryba@kky.zcu.cz',

    windows=[{
            'script': 'memSkel_main.py',
            'icon_resources': [(1, 'data/icons/multiIcon.ico')]
             }],
    #data_files = matplotlib.get_py2exe_datafiles(),
    data_files=data_files,

    options={
        'py2exe': {
            'dist_dir': dist_dir,
            'includes': includes,
            'excludes': excludes,
            'compressed': True, #little bit slower, but smaller distribution
        }
    }
)