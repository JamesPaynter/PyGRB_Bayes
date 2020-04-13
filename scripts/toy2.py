import numpy as np
import pandas as pd
from astropy.io import fits



file = 'stte_list_3770.fits.gz'
file = 'tte_bfits_3770.fits.gz'


with fits.open(file) as hdu_list:
    print(hdu_list[0])
