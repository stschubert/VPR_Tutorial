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
