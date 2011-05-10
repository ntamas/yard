#!/usr/bin/env python

from setuptools import setup
import yard

setup(name='yard',
      version=yard.__version__,
      author='Tamas Nepusz',
      author_email='tamas@cs.rhul.ac.uk',
      url='http://github.com/ntamas/yard',
      description='Yet another ROC curve drawer',
      long_description=yard.__doc__,
      packages=['yard', 'yard.scripts'],
      provides=['yard'],
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
