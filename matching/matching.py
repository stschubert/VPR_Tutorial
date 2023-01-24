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
from scipy.stats import norm


################################################################################
# find the best match per query in S of size |DB|x|Q|
def best_match_per_query(S):
    j = np.argmax(S, axis=0)
    i = np.int64(range(len(j)))

    M = np.zeros_like(S, dtype='bool')
    M[i, j] = True

    return M


################################################################################
# thresholding of S
# automatic thresholding based on Eq. 2-4 in Schubert, S., Neubert, P. & Protzel, P. (2021). Beyond ANN: Exploiting Structural Knowledge for Efficient Place Recognition. In Proc. of Intl. Conf. on Robotics and Automation (ICRA). DOI: 10.1109/ICRA48506.2021.9561006
def thresholding(S, thresh):
    if thresh == 'auto':
        mu = np.median(S)
        sig = np.median(np.abs(S - mu)) / 0.675
        thresh = norm.ppf(1 - 1e-6, loc=mu, scale=sig)

    M = S >= thresh

    return M
