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
from evaluation.createPR import createPR
from evaluation import show_correct_and_wrong_matches
from matching import matching
from feature_aggregation.hdc import hdc
from load_dataset import gardenspoint
from compute_local_descriptors import compute_delf
import numpy as np

from matplotlib import pyplot as plt
plt.ion()


# load dataset
print('===== Load dataset')
db_imgs, q_imgs, GThard, GTsoft = gardenspoint()

# compute local descriptors
print('===== Compute local DELF descriptors')
db_D = compute_delf(db_imgs)
q_D = compute_delf(q_imgs)

# feature aggregation, i.e., local->holistic descriptors
print('===== Compute holistic HDC-DELF descriptors')
db_D_holistic = hdc(db_D).compute_holistic()
q_D_holistic = hdc(q_D).compute_holistic()

# normalize descriptors and compute S-matrix
print('===== Compute cosine similarities S')
db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic , axis=1, keepdims=True)
q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic , axis=1, keepdims=True)
S = np.matmul(db_D_holistic , q_D_holistic.transpose())

# matching decision making
print('===== Match images')

# best match per query -> Single-best-match VPR
M1 = matching.best_match_per_query(S)

# thresholding -> Multi-match VPR
M2 = matching.thresholding(S, 'auto')
TP = np.argwhere(M2 & GThard)  # true positives
FP = np.argwhere(M2 & ~GTsoft)  # false positives

# evaluation
print('===== Evaluation')
# show correct and wrong image matches
show_correct_and_wrong_matches.show(
    db_imgs, q_imgs, TP, FP)  # show random matches
plt.title('Examples for correct and wrong matches from S>=thresh')

# show M's
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(M1)
ax1.axis('off')
ax1.set_title('Best match per query')
ax2 = fig.add_subplot(122)
ax2.imshow(M2)
ax2.axis('off')
ax2.set_title('Thresholding S>=thresh')

# PR-curve
P, R = createPR(S, GThard, GTsoft)
plt.figure()
plt.plot(R, P)
plt.xlim(0, 1), plt.ylim(0, 1.01)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Result on GardensPoint day_right--night_right')
plt.grid('on')
plt.draw()

# area under curve (AUC)
AUC = np.trapz(P, R)
print('\n===== AUC (area under curve):', AUC, '\n')
