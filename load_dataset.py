#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================
#
import os
import urllib.request
import zipfile
from glob import glob
from PIL import Image
import numpy as np
from scipy.signal import convolve2d


################################################################################
def gardenspoint():

    print('===== Load dataset GardensPoint day_right--night_right')

    # download images if necessary
    fn = 'GardensPoint_Walking.zip'
    url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn
    destination = 'images/GardensPoint/'

    if not os.path.exists(destination):
        print('===== GardensPoint dataset does not exist. Download to ' +
              destination + '...')

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)

    # load images
    fns_db = glob(destination + 'day_right/*.jpg')
    fns_q = glob(destination + 'night_right/*.jpg')
    fns_db.sort()
    fns_q.sort()

    imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
    imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

    # create ground truth
    GThard = np.eye(len(imgs_db)).astype('bool')
    GTsoft = convolve2d(GThard.astype('int'),
                        np.ones((17, 1), 'int'), mode='same').astype('bool')

    return imgs_db, imgs_q, GThard, GTsoft
