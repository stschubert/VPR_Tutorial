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
import torch
from torchvision import transforms

from .feature_extractor import FeatureExtractor


class AlexNetConv3Extractor(FeatureExtractor):
    def __init__(self, nDims: int = 4096):
        self.nDims = nDims
        # load alexnet    
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

        # select conv3
        self.model = self.model.features[:7]

        # preprocess images
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if torch.cuda.is_available():
            self.model.to('cuda')


    def compute_features(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        imgs_torch = [self.preprocess(img) for img in imgs]
        imgs_torch = torch.stack(imgs_torch, dim=0)

        if torch.cuda.is_available():
            imgs_torch = imgs_torch.to('cuda')

        with torch.no_grad():
            output = self.model(imgs_torch)

        output = output.to('cpu').numpy()
        Ds = output.reshape([len(imgs), -1])

        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
        Proj = Proj / np.linalg.norm(Proj , axis=1, keepdims=True)

        Ds = Ds @ Proj

        return Ds
