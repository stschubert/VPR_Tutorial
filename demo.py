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
import configparser
import os

from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches
from matching import matching
from datasets.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset, HTT_example
import numpy as np

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=str, default='HDC-DELF', choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD', 'CosPlace', 'EigenPlaces', 'SAD'], help='Select descriptor (default: HDC-DELF)')
    parser.add_argument('--dataset', type=str, default='HTT_example', choices=['GardensPoint', 'StLucia', 'SFU', 'HTT_example'], help='Select dataset (default: GardensPoint)')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{args.dataset}', exist_ok=True)
    output_dir = f'results/{args.dataset}'
    print('========== Start VPR with {} descriptor on dataset {}'.format(args.descriptor, args.dataset))

    # load dataset
    print('===== Load dataset')
    if args.dataset == 'GardensPoint':
        dataset = GardensPointDataset()
    elif args.dataset == 'StLucia':
        dataset = StLuciaDataset()
    elif args.dataset == 'SFU':
        dataset = SFUDataset()
    elif args.dataset == 'HTT_example':
        violaiton_scan = '20-ANG2312_00089-ANG2401_00089'
        os.makedirs(f'results/{args.dataset}/{violaiton_scan}', exist_ok=True)
        output_dir = f'results/{args.dataset}/{violaiton_scan}'
        dataset = HTT_example(destination=f'images/HTT_example/{violaiton_scan}')
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)

    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    if args.descriptor == 'HDC-DELF':
        from feature_extraction.feature_extractor_holistic import HDCDELF
        feature_extractor = HDCDELF()
    elif args.descriptor == 'AlexNet':
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
        feature_extractor = AlexNetConv3Extractor()
    elif args.descriptor == 'SAD':
        from feature_extraction.feature_extractor_holistic import SAD
        feature_extractor = SAD()
    elif args.descriptor == 'NetVLAD' or args.descriptor == 'PatchNetVLAD':
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
        from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
        if args.descriptor == 'NetVLAD':
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')
        else:
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        feature_extractor = PatchNetVLADFeatureExtractor(config)
    elif args.descriptor == 'CosPlace':
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
        feature_extractor = CosPlaceFeatureExtractor()
    elif args.descriptor == 'EigenPlaces':
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
        feature_extractor = EigenPlacesFeatureExtractor()
    else:
        raise ValueError('Unknown descriptor: ' + args.descriptor)

    if args.descriptor != 'PatchNetVLAD' and args.descriptor != 'SAD':
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # normalize descriptors and compute S-matrix
        print('===== Compute cosine similarities S')
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic , axis=1, keepdims=True)
        q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic , axis=1, keepdims=True)
        S = np.matmul(db_D_holistic , q_D_holistic.transpose())
    elif args.descriptor == 'SAD':
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # compute similarity matrix S with sum of absolute differences (SAD)
        print('===== Compute similarities S from sum of absolute differences (SAD)')
        S = np.empty([len(imgs_db), len(imgs_q)], 'float32')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db_D_holistic[i]-q_D_holistic[j]
                dim = len(db_D_holistic[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i,j] = -np.sum(np.abs(diff)) / dim
    else:
        print('=== WARNING: The PatchNetVLAD code in this repository is not optimised and will be slow and memory consuming.')
        print('===== Compute reference set descriptors')
        db_D_holistic, db_D_patches = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic, q_D_patches = feature_extractor.compute_features(imgs_q)
        # S_hol = np.matmul(db_D_holistic , q_D_holistic.transpose())
        S = feature_extractor.local_matcher_from_numpy_single_scale(q_D_patches, db_D_patches)

    # show similarity matrix
    fig = plt.figure()
    plt.imshow(S)
    plt.axis('off')
    plt.title('Similarity matrix S')

    # matching decision making
    print('===== Match images')

    # best match per query -> Single-best-match VPR
    M1 = matching.best_match_per_query(S)

    # thresholding -> Multi-match VPR
    M2 = matching.thresholding(S, 0.4)
    TP = np.argwhere(M2 & GThard)  # true positives
    FP = np.argwhere(M2 & ~GTsoft)  # false positives

    # evaluation
    print('===== Evaluation')
    # show correct and wrong image matches
    show_correct_and_wrong_matches.show(
        imgs_db, imgs_q, TP, FP, output_dir=output_dir)  # show random matches

    # show M's
    fig, axs = plt.subplots(3)
    axs[0].imshow(GThard)
    axs[0].axis('off')
    axs[0].set_title('GT')
    axs[1].imshow(M1)
    axs[1].axis('off')
    axs[1].set_title('Best match per query')
    axs[2].imshow(M2)
    axs[2].axis('off')
    axs[2].set_title('Thresholding S>=thresh')
    fig.savefig(f'{output_dir}/M.png')

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
    plt.savefig(f'{output_dir}/PR.png')
    # area under curve (AUC)
    AUC = np.trapz(P, R)
    print(f'\n===== AUC (area under curve): {AUC:.3f}')

    # maximum recall at 100% precision
    maxR = recallAt100precision(S, GThard, GTsoft, matching='multi', n_thresh=100)
    print(f'\n===== R@100P (maximum recall at 100% precision): {maxR:.2f}')

    # recall at K
    
    RatK = {}
    K_val = [1, min((5, S.shape[0])), min((10, S.shape[0]))]
    for K in K_val:
        RatK[K] = recallAtK(S, GThard, GTsoft, K=K)

    print(f'\n===== recall@K (R@K) -- R@1: {RatK[K_val[0]]:.3f}, R@5: {RatK[K_val[1]]:.3f}, R@10: {RatK[K_val[2]]:.3f}')

    plt.show()


if __name__ == "__main__":
    main()
