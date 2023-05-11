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
from scipy.linalg import orth
from typing import Union, Dict, List, Tuple


class HDC(object):
    """
    A class for implementing Hyperdimensional Computing (HDC).
    """

    def __init__(self, Ds: List[Dict[str, np.ndarray]], nDims: int = 4096, nFeat: int = 200, nX: int = 5, nY: int = 7):
        """
        Initializes the HDC object with the given parameters.

        Args:
            Ds (List[Dict[str, np.ndarray]]): A list of dictionaries containing
                descriptors.
            nDims (int, optional): The number of dimensions for the HDC vectors.
                Defaults to 4096.
            nFeat (int, optional): The number of features. Defaults to 200.
            nX (int, optional): The number of HDC vectors for the X-axis. Defaults
                to 5.
            nY (int, optional): The number of HDC vectors for the Y-axis. Defaults
                to 7.
        """
        indim = Ds[0]['descriptors'].shape[1]
        self.nDims = nDims
        self.nFeat = nFeat
        self.nX = nX
        self.nY = nY
        self.Ds = Ds

        # random projection matrix for descriptors
        rng = np.random.default_rng(123)
        self.Proj = rng.standard_normal((indim, self.nDims), 'float32')
        self.Proj = orth(self.Proj.transpose()).transpose()

        # hdc vectors for poses
        rng = np.random.default_rng(456)
        self.X = (1. - 2. * (rng.random((nX, nDims), 'float32') > 0.5)
                  ).astype('float32')
        self.posX = np.linspace(0., 1., nX)

        rng = np.random.default_rng(789)
        self.Y = (1. - 2. * (rng.random((nY, nDims), 'float32') > 0.5)
                  ).astype('float32')
        self.posY = np.linspace(0., 1., nY)


    def compute_holistic(self) -> np.ndarray:
        """
        Computes the holistic HDC descriptors for each entry in self.Ds.

        Returns:
            np.ndarray: A two-dimensional array with the shape (len(self.Ds), self.nDims),
                containing the holistic HDC descriptors for each entry in self.Ds.
        """
        Y = np.zeros((len(self.Ds), self.nDims), 'float32')

        for i in range(len(self.Ds)):
            # load i-th descriptor
            D = self.Ds[i]

            # standardize local descriptors
            D['descriptors'] = D['descriptors'] @ self.Proj
            D['descriptors'] = self.__STD(D['descriptors'])

            # compute holistic HDC-descriptor
            Y[i, :] = self.__bundleLocalDescriptorsIndividually(D)

        return Y


    def __bundleLocalDescriptorsIndividually(self, D: Dict[str, Union[np.ndarray, int]]) -> np.ndarray:
        """
        Binds each local descriptor to its pose and bundles them to compute the holistic
        HDC descriptor for a given input dictionary D.

        Args:
            D (Dict[str, Union[np.ndarray, int]]): A dictionary containing the following keys:
                - 'descriptors': A two-dimensional array containing the local descriptors.
                - 'keypoints': A two-dimensional array containing the keypoints.
                - 'imheight': An integer representing the image height.
                - 'imwidth': An integer representing the image width.

        Returns:
            np.ndarray: A one-dimensional array of length self.nDims, containing the
                holistic HDC descriptor for the input dictionary D.
        """        
        desc = D['descriptors']
        keypoints = D['keypoints']
        h = D['imheight']
        w = D['imwidth']
        nDims = self.nDims

        # encode poses
        PV = self.__encodePosesHDCconcatMultiAttractor(
            keypoints, h, w, nDims)

        # bind each descriptor to its pose and bundle
        H = np.sum(desc * PV, 0)
        return H


    def __encodePosesHDCconcatMultiAttractor(self, P: np.ndarray, h: int, w: int, nDims: int) -> np.ndarray:
        """
        Encodes the poses of keypoints using the HDC multi-attractor approach.

        Args:
            P (np.ndarray): A two-dimensional array containing the keypoints.
            h (int): The image height.
            w (int): The image width.
            nDims (int): The number of dimensions for the HDC vectors.

        Returns:
            np.ndarray: A two-dimensional array with the shape (P.shape[0], nDims),
                containing the encoded HDC poses for the keypoints.
        """        
        # relative poses for keypoints
        xr = P[:, 1] / w
        yr = P[:, 0] / h

        PV = np.zeros((P.shape[0], nDims), 'float32')
        for i in range(xr.size):
            # find attractors and split index
            Xidx1, Xidx2, XsplitIdx = self.__findAttractorsAndSplitIdx(
                xr[i], self.posX, nDims)
            Yidx1, Yidx2, YsplitIdx = self.__findAttractorsAndSplitIdx(
                yr[i], self.posY, nDims)

            # apply
            xVec = np.concatenate(
                (self.X[Xidx1, :XsplitIdx], self.X[Xidx2, XsplitIdx:]))
            yVec = np.concatenate(
                (self.Y[Yidx1, :YsplitIdx], self.Y[Yidx2, YsplitIdx:]))

            # combine
            PV[i, :] = xVec * yVec

        return PV


    def __findAttractorsAndSplitIdx(self, p: float, pos: np.ndarray, nDims: int) -> Tuple[int, int, int]:
        """
        Finds the two closest attractors to the given position and computes the
        split index for combining the HDC vectors.

        Args:
            p (float): The position of the point in question.
            pos (np.ndarray): A one-dimensional array containing the attractor positions.
            nDims (int): The number of dimensions for the HDC vectors.

        Returns:
            Tuple[int, int, int]: A tuple containing:
                - The index of the first closest attractor.
                - The index of the second closest attractor.
                - The split index for combining the HDC vectors.
        """
        # find closest vectors
        idx = np.argpartition(abs(pos-p), 2)[:2]
        idx.sort()
        idx1, idx2 = idx

        # find weighting of both
        d1 = abs(pos[idx1]-p)
        d2 = abs(pos[idx2]-p)
        w = d2 / (d1 + d2)

        # compute index from weighting
        splitIdx = round(w*nDims)

        # return indices
        return idx1, idx2, splitIdx


    def __STD(self, D: np.ndarray) -> np.ndarray:
        """
        Standardizes the input descriptors by subtracting the mean and dividing by the standard deviation.

        Args:
            D (np.ndarray): A two-dimensional array containing the descriptors to be standardized.

        Returns:
            np.ndarray: A two-dimensional array containing the standardized descriptors.
        """

        mu = D.mean(0)
        sig = D.std(0)

        D = (D-mu) / sig

        return D
