PyGRB Bayes
===========
.. inclusion-marker-one-liner-start
A GRB light-curve analysis package.

.. inclusion-marker-one-liner-end





Directory Structure
-------------------


| ├── PyGRB_Bayes            # The code is in this folder and sub-folders
| ├── data                   # Data downloaded by the software will be placed in this folder (initially empty)
| ├── docs                   # Documentation files
| ├── products               # All created data-products will be saved in this folder (initially empty)
| ├── scripts                # My scripts which use this program (may be deleted later)
| ├── tests                  # Unit tests
| ├── .codecov.yml
| ├── .gitignore
| ├── .travis.yml
| ├── LICENSE
| ├── README.md
| ├── requirements.txt
| └── setup.py


.. inclusion-marker-what-it-does-start
What it does
------------
Code to download GRB light-curves from internet archives (at the moment only BATSE implemented). The code is then able to create light-curves from either pre-binned data or time-tagged photon-event data. Light-curves may then be fitted with with pulse models, and further analysed.

.. inclusion-marker-what-it-does-end



.. inclusion-marker-pulse-types-start
Pulse types
-----
Description of GRB pulse phenomenology.

.. figure:: docs/source/images/B_6630__d_NL200__rates.pdf
    :width: 100%
    :align: center
    :alt: a GRB light-curve

.. inclusion-marker-pulse-types-end



.. inclusion-marker-usage-start
Usage
-----
Instructions on how to use the code.

.. inclusion-marker-usage-end
