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

import numpy as np
from typing import List
from abc import abstractmethod

from tqdm.auto import tqdm

from .feature_extractor import FeatureExtractor


class LocalFeatureExtractor(FeatureExtractor):

    @abstractmethod
    def compute_local_features(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        pass


class DELF(LocalFeatureExtractor):
    def __init__(self):
        import tensorflow_hub as hub

        self.delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:

        D_local = self.compute_local_features(imgs)

        return D_local

    def compute_local_features(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        D = []
        for img in tqdm(imgs):
            D.append(self.compute_local_delf_descriptor(img))

        return D

    def compute_local_delf_descriptor(self, img: np.ndarray):
        import tensorflow as tf

        im_height = img.shape[0]
        im_width = img.shape[1]
        img = tf.image.convert_image_dtype(img, tf.float32)

        out = self.delf(image=img,
                score_threshold=tf.constant(0.0),
                image_scales=tf.constant([1.0]),
                max_feature_num=tf.constant(200))

        return {'descriptors': np.array(out['features']),
                'descriptors_pca': np.array(out['descriptors']),
                'scores': np.array(out['scores']),
                'keypoints': np.array(out['locations']),
                'scales': 1./np.array(out['scales']),
                'imheight': np.array(im_height),
                'imwidth': np.array(im_width)
                }
