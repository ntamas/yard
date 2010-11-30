YARD - Yet Another ROC Drawer
=============================

:Author: Tamás Nepusz

This is yet another Python package for drawing ROC curves. It also
lets you draw precision-recall, accumulation and concentrated ROC
(CROC) curves, sensitivity-specificity plots, F-score curves and calculate
the AUC (area under curve) statistics.  The significance of differences
between AUC scores can also be tested using paired permutation tests.

Where to get ``yard``
---------------------

``yard`` has two homes at the moment:

* The `Python package index`_. This page hosts the most recent stable
  version of ``yard``. Since ``yard`` is under heavy development at the
  moment, you might not get all the latest and greatest features of
  ``yard``, but you will most likely find a version here that should
  not collapse even under exceptional circumstances.

* A page on GitHub_. On this page you can follow the development of
  ``yard`` as closely as possible; you can get the most recent
  development version, file bug reports, or even fork the project
  to start adding your own features.

.. _Python package index: http://pypi.python.org/pypi/yard
.. _GitHub: http://github.com/ntamas/yard

Requirements
------------

You will need the following tools to run ``yard``:

* `Python 2.6`_ or later.

* `Matplotlib`_, which is responsible for plotting the curves. If
  you don't have `Matplotlib`_, you can export the points of the
  curves and then use an external plotting tool such as `GNUPlot`_
  to plot them later.

* `NumPy`_ is an optional dependency; some functions will be
  slightly faster if you have `NumPy`_, but ``yard`` should work
  fine without it as well.

.. _Python 2.6: http://www.python.org
.. _Matplotlib: http://matplotlib.sourceforge.net
.. _GNUPlot: http:/www.gnuplot.info
.. _NumPy: http://numpy.scipy.org

Installation
------------

The simplest way to install ``yard`` is by using ``easy_install``::

    $ easy_install yard

This goes to the `Python package index`_ page, fetches the most recent
stable version and installs it, creating two scripts in your path:
``yard-plot`` for plotting and ``yard-significance`` for significance
testing.

If you want the bleeding edge version, you should go to the GitHub_
page, download a ZIP or .tar.gz file, extract it to some directory
and then run the following command::

    $ python setup.py install

Running ``yard``
----------------

``yard`` works with simple tabular flat files, and assumes that the first
row in each file is a header. Each row contains data related to a particular
test example. By default, the first column contains the *expected* outcome
of a binary classifier for a given test example (i.e. whether the example is
positive or negative), while the remaining columns contain the output of
the probabilistic classifiers being tested on the test dataset. The
expected outcome must be positive for positive examples and zero or negative
for negative examples - this means that you may use either zeros and ones
or -1 and 1 for negative and positive test examples, respectively. The
outcomes of the classifiers may be in any range, but they are most frequently
in the interval [0; 1]. The following snippet shows an example input file::

    output  Method1 Method2 Method3
    -1      0.2     0.3     0.6
    -1      0.4     0.15    0.1
    +1      0.7     0.2     0.9
    +1      0.3     0.85    1.0

Columns must be separated by tabs per default, but this can be overridden
with the ``-f`` option on the command line. The actual columns being used
can also be overridden using ``-c``; for instance, if you have the expected
outcome in column 4 and the actual outcomes in columns 1-3, you may use
``-c 4,1-3`` to specify that.

Some usage examples are presented here; for more details, type
``yard-plot --help`` or ``yard-significance --help``.

To show a ROC curve for an arbitrary number of classifiers where the expected
and actual outcomes are defined in ``input_data.txt``::

    $ yard-plot input_data.txt

If the actual outcomes are in columns 3-5, the expected outcome is in
column 6 and the columns are separated by semicolons::

    $ yard-plot -f ';' -c 6,3-5 input_data.txt

To plot precision-recall curves instead of ROC curves and also show the
AUC statistics::

    $ yard-plot -t pr --show-auc input_data.txt

Supported curve types are: ``roc`` for ROC curves (default), ``pr`` for
precision-recall curves, ``croc`` for CROC curves, ``ac`` for accumulation
curves, ``sespe`` for sensitivity-specificity plots, ``fscore`` for
F-score curves.

To use a logarithmic X axis for the ROC curve and use the standard input
instead of a file::

    $ yard-plot -l x

The omission of an input filename instructs ``yard-plot`` to use the standard
input. You may have also used ``-`` in place of the filename to specify that.

To save a ROC curve into a PDF file::

    $ yard-plot -o roc_curve.pdf input_data.txt

You may specify other formats as long as they are supported by Matplotlib::

    $ yard-plot -o roc_curve.ps input_data.txt
    $ yard-plot -o roc_curve.png input_data.txt

The PDF backend also supports multiple plots in separate pages::

    $ yard-plot -t pr -t roc -t croc -o curves.pdf input_data.txt

The figure size, the DPI ratio and the font size can also be adjusted::

    $ yard-plot -o roc_curve.pdf --font-size 8 -s '8cm x 6cm' input_data.txt

To calculate the AUC statistics for multiple curves without plotting them::

    $ yard-auc -t pr -t roc input_data.txt

To test whether the ROC curves of multiple classifiers are significantly
different::

    $ yard-significance input_data.txt

Questions, comments
-------------------

If you have a question or comment about ``yard`` or you think you have
found a bug, feel free to `contact me`_.

.. _contact me: http://www.cs.rhul.ac.uk/home/tamas

Acknowledgments and references
------------------------------

The inclusion of CROC curves and the statistical significance testing
was inspired by the following publication (which also provides more
details on what CROC curves are and why they are more useful than ROC
curves in many cases):

    **A CROC Stronger than ROC: Measuring, Visualizing and Optimizing
    Early Retrieval**.
    S. Joshua Swamidass, Chloe-Agathe Azencott, Kenny Daily and Pierre Baldi.
    *Bioinformatics*, April 2010, doi:10.1093/bioinformatics/btq140

