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

import argparse

from feature_extraction.feature_extractor_local import DELF
from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches
from matching import matching
from datasets.load_dataset import GardensPointDataset
import numpy as np

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=str, default='HDC-DELF', choices=['HDC-DELF', 'AlexNet'], help='Select descriptor')
    parser.add_argument('--dataset', type=str, default='GardensPoint', choices=['GardensPoint'], help='Select dataset')
    args = parser.parse_args()

    # plt.ion()

    # load dataset
    print('===== Load dataset')
    if args.dataset == 'GardensPoint':
        dataset = GardensPointDataset()
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)

    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    if args.descriptor == 'HDC-DELF':
        feature_extractor = DELF()
    elif args.descriptor == 'AlexNet':
        feature_extractor = AlexNetConv3Extractor()
    else:
        raise ValueError('Unknown descriptor: ' + args.descriptor)

    print('===== Compute reference set descriptors')
    db_D_holistic = feature_extractor.compute_features(imgs_db)
    print('===== Compute query set descriptors')
    q_D_holistic = feature_extractor.compute_features(imgs_q)

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
        imgs_db, imgs_q, TP, FP)  # show random matches
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
    P, R = createPR(S, GThard, GTsoft, matching='multi', n_thresh=100)
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
    print(f'\n===== AUC (area under curve): {AUC:.3f}')

    # maximum recall at 100% precision
    maxR = recallAt100precision(S, GThard, GTsoft, matching='multi', n_thresh=100)
    print(f'\n===== R@100P (maximum recall at 100% precision): {maxR:.3f}')

    # recall at K
    Rat1 = recallAtK(S, GThard, GTsoft, K=1)
    Rat5 = recallAtK(S, GThard, GTsoft, K=5)
    Rat10 = recallAtK(S, GThard, GTsoft, K=10)
    print(f'\n===== recall@K (R@K) -- R@1: {Rat1:.3f}, R@5: {Rat5:.3f}, R@10: {Rat10:.3f}')

    plt.show()


if __name__ == "__main__":
    main()
