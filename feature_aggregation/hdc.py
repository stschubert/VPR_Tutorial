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


################################################################################
class hdc:

    ############################################################################
    def __init__(self, Ds, nDims=4096, nFeat=200, nX=5, nY=7):
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

    ############################################################################
    def compute_holistic(self):
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

    ############################################################################
    def __bundleLocalDescriptorsIndividually(self, D):
        desc = D['descriptors']
        keypoints = D['keypoints']
        nFeat = min(self.nFeat, desc.shape[0])
        h = D['imheight']
        w = D['imwidth']
        nDims = self.nDims

        # encode poses
        PV = self.__encodePosesHDCconcatMultiAttractor(
            keypoints, h, w, nDims)

        # bind each descriptor to its pose and bundle
        H = np.sum(desc * PV, 0)
        return H

    ############################################################################
    def __encodePosesHDCconcatMultiAttractor(self, P, h, w, nDims):
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

    ############################################################################
    def __findAttractorsAndSplitIdx(self, p, pos, nDims):
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

    ############################################################################
    def __STD(self, D):
        # perform descriptor standardization
        mu = D.mean(0)
        sig = D.std(0)

        D = (D-mu) / sig

        return D
