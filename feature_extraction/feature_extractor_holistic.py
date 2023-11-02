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

from .feature_extractor import FeatureExtractor


class AlexNetConv3Extractor(FeatureExtractor):
    def __init__(self, nDims: int = 4096):
        import torch
        from torchvision import transforms

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
            print('Using GPU')
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print('Using MPS')
            self.device = torch.device("mps")
        else:
            print('Using CPU')
            self.device = torch.device("cpu")

        self.model.to(self.device)


    def compute_features(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        import torch

        imgs_torch = [self.preprocess(img) for img in imgs]
        imgs_torch = torch.stack(imgs_torch, dim=0)

        imgs_torch = imgs_torch.to(self.device)

        with torch.no_grad():
            output = self.model(imgs_torch)

        output = output.to('cpu').numpy()
        Ds = output.reshape([len(imgs), -1])

        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
        Proj = Proj / np.linalg.norm(Proj , axis=1, keepdims=True)

        Ds = Ds @ Proj

        return Ds


class HDCDELF(FeatureExtractor):
    def __init__(self):
        from .feature_extractor_local import DELF

        self.DELF = DELF() # local DELF descriptor

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        from feature_aggregation.hdc import HDC

        D_local = self.DELF.compute_features(imgs)
        D_holistic = HDC(D_local).compute_holistic()

        return D_holistic


# sum of absolute differences (SAD) [Milford and Wyeth (2012). "SeqSLAM: Visual
# Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights". ICRA.]
class SAD(FeatureExtractor):
    def __init__(self, nPixels: int = 2048, patchLength: int = 8):
        self.nPixels = nPixels # number pixels in downsampled image
        self.patchLength = patchLength # side length of patches for patch normalization

        self.imshapeDownsampled = None

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:

        # determine new image shape to obtain roughly self.nPixels pixels and
        # image dimensions that are a multiple of self.patchLength
        if self.imshapeDownsampled is None:
            [h,w,_] = np.array(imgs[0].shape)

            k = np.sqrt(self.nPixels / (h * w))
            h = np.ceil(k * h)
            h -= np.mod(h, self.patchLength)
            w = np.ceil(k * w)
            w -= np.mod(w, self.patchLength)

            if np.abs(self.nPixels - h*w) > np.abs(self.nPixels - (h+self.patchLength)*(w+self.patchLength)):
                h += self.patchLength
                w += self.patchLength

            self.imshapeDownsampled = [int(h), int(w)]

        # grayscale conversion and downsampling
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(self.imshapeDownsampled),
        ])
        imgs = [np.array(preprocess(img)) for img in imgs]

        # patch normalization
        Ds = [self.__patch_normalize(img).flatten() for img in imgs]
        Ds = np.array(Ds).astype('float32')

        return Ds

    def __patch_normalize(self, img: np.ndarray) -> np.ndarray:
        np.seterr(divide='ignore', invalid='ignore') # ignore potential division by 0

        img = img.astype('float32')
        [h,w] = img.shape
        for i_h in range(h // self.patchLength):
            for i_w in range(w // self.patchLength):
                patch = img[i_h*self.patchLength:(i_h+1)*self.patchLength, i_w*self.patchLength:(i_w+1)*self.patchLength]
                patch_normalized = 255 * ((patch - patch.min()) / (patch.max() - patch.min()))
                patch = patch_normalized.round()
                img[i_h*self.patchLength:(i_h+1)*self.patchLength, i_w*self.patchLength:(i_w+1)*self.patchLength] = patch

        np.seterr(divide='warn', invalid='warn')

        return img
