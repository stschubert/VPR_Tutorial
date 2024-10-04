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
import pandas as pd

from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches
from matching import matching
from datasets.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset, HTT_example
from utils.geoutils import haversine
from utils.imgutils import xywhn2xyxy, xywh2xyxy, xywh2xyxy_test, normalise_xyxy, iou
import numpy as np

from matplotlib import pyplot as plt
import copy

def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=str, default='HDC-DELF', choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD', 'CosPlace', 'EigenPlaces', 'SAD'], help='Select descriptor (default: HDC-DELF)')
    parser.add_argument('--dataset', type=str, default='5-ANG2312_00089-ANG2401_00089', help='Select Hubble dataset')
    parser.add_argument('--iou', type=float, default=None, help='IoU limit on detections. Any below this will not be considered as a match')
    parser.add_argument('--distance', type=float, default=None, help='Distance limit in km. Any violations above this will not be considered as a match')
    parser.add_argument('--output', type=str, default='HTT_example', help='Output directory for results')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{args.output}', exist_ok=True)
    output_dir = f'results/{args.output}'
    print('========== Start VPR with {} descriptor on dataset {}'.format(args.descriptor, args.dataset))

    # load dataset
    print('===== Load dataset')
    violation_scan = args.dataset
    os.makedirs(f'results/HTT_example/{violation_scan}', exist_ok=True)
    output_dir = f'results/HTT_example/{violation_scan}'
    dataset = HTT_example(destination=f'images/HTT_example/{violation_scan}')

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

    fns_db, fns_q = dataset.get_images_names()
    violations_entry_db, violations_entry_q = dataset.get_violations_df()
    S_old = copy.copy(S)

    # Get parameters from command line
    radius_threshold = args.distance
    iou_threshold = args.iou

    for i, fn_db in enumerate(fns_db):
        id_db = fn_db.split('/')[-1].split('.')[0]
        entry_db = violations_entry_db[violations_entry_db['id'] == int(id_db)]

        # Calculate IOU score
        # Why are some normalised and others not?
        if entry_db['box_startx'].values[0]<0 and entry_db['box_starty'].values[0]<0 and entry_db['box_height'].values[0]<0 and entry_db['box_width'].values[0] < 0:
            # TODO: Look into the ordering of these. This extracts correct bbox but I am not sure why...
            xyxy_db = xywhn2xyxy(np.array([[entry_db['box_startx'].values[0], entry_db['box_starty'].values[0], entry_db['box_height'].values[0], entry_db['box_width'].values[0]]]), w=imgs_db[i].shape[0], h=imgs_db[i].shape[1])
        else:
            # TODO: Look into why our data needs this function...
            xyxy_db = xywh2xyxy_test(np.array([[entry_db['box_starty'].values[0], entry_db['box_startx'].values[0], entry_db['box_height'].values[0], entry_db['box_width'].values[0]]]))

        # Normalise against width/height so we can compare
        xyxy_n_db = normalise_xyxy(xyxy_db, imgs_db[i].shape[0], imgs_db[i].shape[1])

        for j, fn_q in enumerate(fns_q):
            id_q = fn_q.split('/')[-1].split('.')[0]
            entry_q = violations_entry_q[violations_entry_q['id'] == int(id_q)]

            # TODO: Set simularity to 0 for all images outside of a certain radius on GPS
            # Look into why this isnt working properly
            if radius_threshold is not None:
                distance = haversine(entry_db['lat'].to_list()[0], entry_db['long'].to_list()[0], entry_q['lat'].to_list()[0], entry_q['long'].to_list()[0])
                if(distance>radius_threshold):
                    S[i, j] = 0

            # TODO: Set simularity to 0 for all IoUs which are less than a certain radius
            # TODO: Look into the ordering of these. This extracts correct bbox but I am not sure why...
            # TODO: Look into why some of these are normalised and some arent...


            # Calculate IOU score
            # Again, why are some normalised and others not?
            if entry_q['box_startx'].values[0]<0 and entry_q['box_starty'].values[0]<0 and entry_q['box_height'].values[0]<0 and entry_q['box_width'].values[0] < 0:
                # TODO: Look into the ordering of these. This extracts correct bbox but I am not sure why...
                xyxy_q = xywhn2xyxy(np.array([[entry_q['box_startx'].values[0], entry_q['box_starty'].values[0], entry_q['box_height'].values[0], entry_q['box_width'].values[0]]]), w=imgs_q[i].shape[0], h=imgs_q[i].shape[1])
            else:
                # TODO: Look into why our data needs this function...
                xyxy_q = xywh2xyxy_test(np.array([[entry_q['box_starty'].values[0], entry_q['box_startx'].values[0], entry_q['box_height'].values[0], entry_q['box_width'].values[0]]]))
            
            # Normalise against width/height so we can compare
            xyxy_n_q = normalise_xyxy(xyxy_q, imgs_q[i].shape[0], imgs_q[i].shape[1])

            if iou_threshold is not None:
                iou_score = iou(xyxy_n_db, xyxy_n_q)
                if iou_score < iou_threshold:
                    S[i, j] = 0


            # TODO: Then look through all image bboxs, run some kind of feature extraction on the violation? Then use this for final matching?

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
    M2 = matching.thresholding(S, 'auto')
    
    # Combining: use best match only when above threshold
    M3 = np.logical_and(M1, M2)

    TP = np.argwhere(M3 & GThard)  # true positives
    FP = np.argwhere(M3 & ~GTsoft)  # false positives

    # evaluation
    print('===== Evaluation')
    # show correct and wrong image matches
    show_correct_and_wrong_matches.show(
        imgs_db, imgs_q, TP, FP, output_dir=output_dir)  # show random matches

    # show M's
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(GThard)
    axs[0, 0].axis('off')
    axs[0, 0].set_title('GT')
    axs[0, 1].imshow(M1)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Best match per query')
    axs[1, 0].imshow(M2)
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Thresholding S >= thresh')
    axs[1, 1].imshow(M3)
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Best match >= thresh')
    fig.savefig(f'{output_dir}/M.png')

    # PR-curve
    P, R = createPR(S, GThard, GTsoft, matching='multi', n_thresh=100)
    plt.figure()
    plt.plot(R, P)
    plt.xlim(0, 1), plt.ylim(0, 1.01)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Result on {violation_scan}')
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

    output_df = pd.DataFrame({'TP': [len(TP)], 'FP': [len(FP)], 'AUC': [AUC], 'maxR': [maxR], 'R@1': [RatK[K_val[0]]], 'R@5': [RatK[K_val[1]]], 'R@10': [RatK[K_val[2]]]})
    output_df.to_csv(f'{output_dir}/results.csv')

if __name__ == "__main__":
    main()
