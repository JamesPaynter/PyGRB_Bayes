import os
import urllib
import numpy as np

from bs4 import BeautifulSoup
from os.path import join, exists

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

class GetBATSEBurst():
    ''' A class to download BATSE bursts. '''

    def __init__(self, trigger, datatype):
        ''' '''

        datatypes = {   'tte'    : 'tte',
                        'discsc' : 'discsc',
                    }

        self._base_string = 'https://heasarc.gsfc.nasa.gov/FTP/compton/data/batse/trigger/'
        self._datatype    = datatype
        self._trigger     = trigger
        self._root        = f'data/BATSE/{datatype}/'
        self._file_name   = f'{datatypes[datatype]}_bfits_{trigger}.fits.gz'
        self._url         = self._make_url()

        mkdir(self._root)
        self.download_file()

    def _make_url(self):
        ''' Takes the trigger number and find the correct subfolder on the NASA
            server. Triggers are separated into folders of 200.
        '''
        zeroes = '0' * (5 - len(str(self._trigger)))
        trigger_5 = f'{zeroes}{self._trigger}'
        s = self._trigger - self._trigger % 200 + 1
        t = '0' * (5 - len(str(s)))
        start = f'{t}{s}'
        e = s + 199
        t = '0' * (5 - len(str(e)))
        stop = f'{t}{e}'
        return f'{self._base_string}{start}_{stop}/{trigger_5}_burst'

    def download_file(self):
        ''' Requests the file from the NASA server. '''
        path = join(self._root, self._file_name)
        if not exists(path):
            remote = f'{self._url}/{self._file_name}'
            try:
                urllib.request.urlretrieve(remote, path)
            except:
                print(  f'The file:  << {self._file_name} >>  '
                        f'does not exist at the specified url:\n'
                        f'{remote}' )
        self.path = path

if __name__ == '__main__':
    pass
