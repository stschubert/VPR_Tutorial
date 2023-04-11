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


def createPR(S_in, GThard, GTsoft, matching='multi', n_thresh=100):
    """
    Calculates the precision and recall at n_thresh equally spaced threshold values
    for a given similarity matrix S_in and ground truth matrices GThard and GTsoft for
    single-best-match VPR or multi-match VPR.

    The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
    same shape.
    The matrices GThard and GTsoft should be binary matrices, where the entries are
    only zeros or ones.
    The matrix S_in should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The string matching should be set to either "single" or "multi" for single-best-
    match VPR or multi-match VPR.
    The integer n_tresh controls the number of threshold values and should be >1.
    """

    assert (S_in.shape == GThard.shape and S_in.shape == GTsoft.shape),"S_in, GThard and GTsoft must have the same shape"
    assert (S_in.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (matching in ['single', 'multi']),"matching should contain one of the following strings: [single, multi]"
    assert (n_thresh > 1),"n_thresh must be >1"

    # ensure logical datatype in GT and GTsoft
    GT = GThard.astype('bool')
    GTsoft = GTsoft.astype('bool')

    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S_in.copy()
    S[GTsoft & ~GT] = S.min()

    # single-best-match or multi-match VPR
    if matching == 'single':
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT.any(0))

        # GT-values for best match per query (i.e., per column)
        GT = GT[np.argmax(S, axis=0), np.arange(GT.shape[1])]

        # similarities for best match per query (i.e., per column)
        S = np.max(S, axis=0)

    elif matching == 'multi':
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT) # ground truth positives

    # init precision and recall vectors
    R = [0, ]
    P = [1, ]

    # select start and end treshold
    startV = S.max()  # start-value for treshold
    endV = S.min()  # end-value for treshold

    # iterate over different thresholds
    for i in np.linspace(startV, endV, n_thresh):
        B = S >= i  # apply threshold

        TP = np.count_nonzero(GT & B)  # true positives
        FP = np.count_nonzero((~GT) & B)  # false positives

        P.append(TP / (TP + FP))  # precision
        R.append(TP / GTP)  # recall

    return P, R
