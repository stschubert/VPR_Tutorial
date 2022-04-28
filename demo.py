from evaluation.createPR import createPR
from feature_aggregation.hdc import hdc
from load_dataset import gardenspoint
from compute_local_descriptors import compute_delf
import numpy as np

from matplotlib import pyplot as plt
plt.ion()


# load dataset
db_imgs, q_imgs, GThard, GTsoft = gardenspoint()

# compute local descriptors
db_D = compute_delf(db_imgs)
q_D = compute_delf(q_imgs)

# feature aggregation, i.e., local->holistic descriptors
db_D_holistic = hdc(db_D).compute_holistic()
q_D_holistic = hdc(q_D).compute_holistic()

# compute S-matrix
S = db_D_holistic @ q_D_holistic.transpose()

# evaluate
P, R = createPR(S, GThard, GTsoft)
AUC = np.trapz(P, R)

plt.plot(R, P)
plt.draw()
print('AUC:', AUC)
