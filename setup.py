#!/usr/bin/env python

from setuptools import setup, find_packages

__version__ = None
exec(open("yard/version.py").read())

long_description = """\
YARD - Yet Another ROC Drawer
=============================

This is yet another Python package for drawing ROC curves. It also
lets you draw precision-recall, accumulation and concentrated ROC
(CROC) curves and calculate the AUC (area under curve) statistics.
The significance of differences between AUC scores can also be
tested using paired permutation tests.

You may also be interested in CROC_, a similar package on the
Python Package Index that implements ROC curves. ``yard`` was developed
independently from CROC_, but several features of CROC have inspired
similar ones in ``yard``.

.. _CROC: http://pypi.python.org/pypi/CROC
"""

setup(name='yard',
      version=__version__,
      author='Tamas Nepusz',
      author_email='tamas@cs.rhul.ac.uk',
      url='http://github.com/ntamas/yard',
      description='Yet another ROC curve drawer',
      long_description=long_description,
      packages=find_packages(exclude=["test"]),
      provides=['yard'],
      test_suite="test",
      keywords='roc curve statistics mathematics machine learning auc',
      license='MIT License',
      entry_points={
          "console_scripts": [
              "yard-auc = yard.scripts.auc:main",
              "yard-plot = yard.scripts.plot:main",
              "yard-significance = yard.scripts.significance:main"
          ]
      },
      extras_require={
          "plotting": ["matplotlib >= 0.99"]
      },
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Utilities']
)
