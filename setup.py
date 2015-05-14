import numpy as np
from setuptools import setup, Extension
import Cython
from Cython.Distutils import build_ext

ext = Extension('holmes_rubin',
        language='c++',
        sources=['holmes_rubin.pyx'],
        include_dirs=[np.get_include()])

setup(name='holmes_rubin',
      author='Robert McGibbon',
      author_email='rmcgibbo@gmail.com',
      url='https://github.com/rmcgibbo/holmes_rubin',
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      zip_safe=False,
      ext_modules=[ext],
      cmdclass={'build_ext': build_ext})
