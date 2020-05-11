=====
PyGRB_Bayes
=====
A GRB light-curve analysis package.





&nbsp;
#### Directory Structure

```
.
├── PyGRB_Bayes             # The code is in this folder and sub-folders
├── data                    # Data downloaded by the software will be placed in this folder (initially empty)
├── docs                    # Documentation files
├── products                # All created data-products will be saved in this folder (initially empty)
├── scripts                 # My scripts which use this program (may be deleted later)
├── tests                   # Test scripts
├── .codecov.yml            
├── .gitignore               
├── .travis.yml             
├── LICENSE                 
├── README.md
├── requirements.txt
└── setup.py
```



&nbsp;
#### What it does
Code to download GRB light-curves from internet archives (at the moment only 
BATSE implemented). The code is then able to create light-curves from either pre-binned data or time-tagged photon-event data. Light-curves may then be fitted with with pulse models, and further analysed.
