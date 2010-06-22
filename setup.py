#!/usr/bin/env python

from setuptools import setup
import yard

setup(name='yard',
      version='0.1',
      author='Tamas Nepusz',
      author_email='tamas@cs.rhul.ac.uk',
      url='http://github.com/ntamas/yard',
      description='Yet another ROC curve drawer',
      long_description=yard.__doc__,
      packages=['yard'],
      provides=['yard'],
      keywords='roc curve statistics mathematics machine learning auc',
      license='GNU General Public License v3',
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence', 
                   'Topic :: Scientific/Engineering :: Bio-Informatics', 
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Utilities']
)
