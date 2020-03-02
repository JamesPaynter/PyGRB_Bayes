# import requests


import urllib
import numpy as np
import time
from bs4 import BeautifulSoup
from os.path import join, exists
import time


class GetBursts():
    '''A class to scrape all the desired bursts off the internet.'''

    def __init__(self, test = True, filetype = None):
        '''The stuff you need to initialise this bad boy.
           Uses https link rather than ftp because it is easier to
           code (for me).'''

        self._base_string = 'https://cossc.gsfc.nasa.gov/FTP/compton/data/batse/trigger/'
        self._test        = test
        self._filetype    = filetype

    def download_file(self, file_name = None, trigger_address = None):

        '''This function downloads a single file from
           the BATSE ftp site. There is a wait function
           in here so I don't get banned.'''

        root_url = 'ftp://legacy.gsfc.nasa.gov/compton/data/batse/trigger/'
        root = 'BATSE_Data_discsc'
        ### if the root directory doesn't exist the program crashes
        ### should make it just create the directory...
        resource = join(root, file_name)
        if not exists(resource):
            wait = np.random.randint(1, 4)
            print('Waiting %i seconds...' % wait)
            time.sleep(wait)
            remote = root_url + trigger_address + file_name
            try:
                urllib.request.urlretrieve(remote, resource)
            except:
                print('This file (%s) does not exist at the specified url\n%s' % (file_name, remote))
                pass
        return resource


    def get_trigger_paths(self):
        '''Get's all the trigger paths of the NASA website.
           Then downloads them, given a file type.

           .gz files on the HTTPS website are corrupted...
           can only download from the ftp links.
           '''

        counter = 1

        ### randomises over the 41 trigger folders
        range_arr = np.random.choice(41, 41, replace=False)
        print(range_arr)
        ### test to not spam the website
        if self._test == True:
            range_arr = [range_arr[0]]
            print(range_arr)

        for i in range_arr:
            counter = i * 200
            # create the url for the trigger folders (separated into groups of 200)
            first_bound = '0' * (5 - len(str(counter))) + str(counter + 1)
            second_bound = '0' * (5 - len(str(counter + 200))) + str(counter + 200)
            indent_level_one = first_bound + '_' + second_bound + '/'
            url_level_one = self._base_string + indent_level_one
            print(url_level_one)

            page = urllib.request.urlopen(url_level_one)
            soup = BeautifulSoup(page, 'html.parser')
            bursts = []
            for link in soup.find_all('a'):
                kk = link.get('href')
                if 'burst' in kk:
                    bursts.append(kk)

            ### randomises over the bursts in the folder
            rand_bursts = np.random.choice(bursts, len(bursts), replace = False)
            if self._test == True:
                rand_bursts = [rand_bursts[0]]
            print(rand_bursts)
            for j in rand_bursts:
                file_num = int(j.replace('_burst/', ''))
#                 file = '%i_sum.gif' % file_num
                file = 'discsc_bfits_%i.fits.gz' % file_num
                indent_level_two = j
                url_level_two    = url_level_one + indent_level_two
                print(indent_level_two, file)
                trig_loc = indent_level_one + indent_level_two
                self.download_file(file_name = file, trigger_address = trig_loc)


            print('done.')



aaa = GetBursts(test = False)
aaa.get_trigger_paths()

print('Super done.')
