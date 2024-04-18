# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:20:04 2024

@author: chels
"""

# Files not yet downloaded
# Files already downloaded
# Files RECENTLY downloaded

# Objective:
    # Remove items in recently downloaded from not yet downloaded
    # Add items in recently downloaded to already downloaded

import glob
import os

pathto_recentdwnld = 'C:/Users/chels/.keras/datasets/costar_block_stacking_dataset_v0.4/johns_hopkins_costar_dataset/blocks_only/*.h5f'

pathto_recentdwnld_file = 'C:/Users/chels/.keras/datasets/costar_block_stacking_dataset_v0.4/recentdownloads.txt'

pathto_alreadydwnld_file = 'C:/Users/chels/.keras/datasets/costar_block_stacking_dataset_v0.4/[DONE]costar_block_stacking_dataset_v0.4_blocks_only_success_only_train_files.txt'

pathto_notyetdwnld_file = 'C:/Users/chels/.keras/datasets/costar_block_stacking_dataset_v0.4/costar_block_stacking_dataset_v0.4_blocks_only_success_only_train_files.txt'

#
trunc_directory = 'johns_hopkins_costar_dataset/blocks_only/'
filelist = list(map(os.path.basename, glob.glob(pathto_recentdwnld)))
recent_dwnlds_list = list(os.path.join(trunc_directory, x) for x in filelist)
recent_dwnlds_dirfiles = set(recent_dwnlds_list)

recent_dwnlds_files = set(glob.glob(pathto_recentdwnld))

print('number of recent downloads: ', len(recent_dwnlds_dirfiles))
      
with open(pathto_alreadydwnld_file, 'r') as all_done, open(pathto_notyetdwnld_file, 'r') as not_yet:
    print('number of files still to be downloaded: ', len(not_yet.readlines()))
    print('number of files already downloaded: ', len(all_done.readlines()))
all_done.close()
not_yet.close()

with open(pathto_alreadydwnld_file, 'a') as all_done, open(pathto_notyetdwnld_file, 'r+') as not_yet:
    lines = not_yet.readlines()
    not_yet.seek(0); #all_done.seek(0)
    not_yet.truncate(); #all_done.truncate()
    
    for line in lines:
        # Add items in recently downloaded to already downloaded
        if any(file in line for file in recent_dwnlds_dirfiles):
            all_done.write(line)

        # Remove items in recently downloaded from not yet downloaded
        # i.e. rewrite all lines in the not_yet file EXCEPT those in recents
        else:
            not_yet.write(line)

all_done.close()
not_yet.close()

    
with open(pathto_alreadydwnld_file, 'r') as all_done, open(pathto_notyetdwnld_file, 'r') as not_yet:
    print('number of files still to be downloaded (after deletion): ', len(not_yet.readlines()))
    print('number of files already downloaded (after addition): ', len(all_done.readlines()))               
all_done.close()
not_yet.close()