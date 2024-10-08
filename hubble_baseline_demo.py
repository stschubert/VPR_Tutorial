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
from utils.imgutils import xywhn2xyxy, xywh2xyxy, xywh2xyxy_test, normalise_xyxy, iou, resize_bbox
import numpy as np

from matplotlib import pyplot as plt
import copy
import cv2
from scipy.spatial.distance import cosine
import glob

def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=str, default='HDC-DELF', choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD', 'CosPlace', 'EigenPlaces', 'SAD'], help='Select descriptor (default: HDC-DELF)')
    parser.add_argument('--dataset', type=str, default='1-ANG2312_00668-ANG2401_00668', help='Select Hubble dataset')
    parser.add_argument('--iou', type=float, default=None, help='IoU limit on detections. Any below this will not be considered as a match')
    parser.add_argument('--distance', type=float, default=None, help='Distance limit in km. Any violations above this will not be considered as a match')
    parser.add_argument('--output', type=str, default='HTT_example', help='Output directory for results')
    parser.add_argument('--local_descriptor', type=str, default='None', choices=['HDC-DELF'], help='Run an additional feature comparison on image bounding boxes using the descriptor (default: None)')
    parser.add_argument('--local_descriptor_normalised', type=str, default='None', choices=['HDC-DELF'], help='Run an additional feature comparison on image bounding boxes, but trying to use the same bounding box size between the image (default: None)')
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
    

    # Get parameters from command line
    radius_threshold = args.distance
    iou_threshold = args.iou
    
    # Get stuff we need for future checks (xyxy, violation images, etc)
    # TODO: do this in a way that doesnt add more loops on the script
    # For now I am prioritising keeping the script easy to understand, can optimise later

    xyxy_dbs = []
    xyxy_n_dbs = []
    violation_imgs_db = []

    xyxy_qs = []
    xyxy_n_qs = []
    violation_imgs_q = []
    
    for i, fn_db in enumerate(fns_db):
        id_db = fn_db.split('/')[-1].split('.')[0]
        entry_db = violations_entry_db[violations_entry_db['id'] == int(id_db)]

        # Calculate xyxy and append
        if entry_db['box_startx'].values[0] <= 1 and entry_db['box_starty'].values[0] <= 1 and entry_db['box_height'].values[0] <= 1 and entry_db['box_width'].values[0] <= 1:
            # TODO: Look into the ordering of these. This extracts correct bbox but I am not sure why...
            xyxy_db = xywhn2xyxy(np.array([[entry_db['box_startx'].values[0], entry_db['box_starty'].values[0], entry_db['box_height'].values[0], entry_db['box_width'].values[0]]]), w=imgs_db[i].shape[0], h=imgs_db[i].shape[1])
        else:
            # TODO: Look into why our data needs this function...
            xyxy_db = xywh2xyxy_test(np.array([[entry_db['box_starty'].values[0], entry_db['box_startx'].values[0], entry_db['box_height'].values[0], entry_db['box_width'].values[0]]]))

        xyxy_dbs.append(xyxy_db)

        # Normalise against width/height so we can compare if the images have different sizes between batches
        xyxy_n_db = normalise_xyxy(xyxy_db, imgs_db[i].shape[0], imgs_db[i].shape[1])
        xyxy_n_dbs.append(xyxy_n_db)
        
        # Extract violation image for testing local features
        violation_imgs_db.append(imgs_db[i][int(xyxy_db[0][0]):int(xyxy_db[0][2]), int(xyxy_db[0][1]):int(xyxy_db[0][3])])

    for i, fn_q in enumerate(fns_q):
        id_q = fn_q.split('/')[-1].split('.')[0]
        entry_q = violations_entry_q[violations_entry_q['id'] == int(id_q)]

        # Calculate xyxy and append
        if entry_q['box_startx'].values[0] <= 1 and entry_q['box_starty'].values[0] <= 1 and entry_q['box_height'].values[0] <= 1 and entry_q['box_width'].values[0] <= 1:
            # TODO: Look into the ordering of these. This extracts correct bbox but I am not sure why...
            xyxy_q = xywhn2xyxy(np.array([[entry_q['box_startx'].values[0], entry_q['box_starty'].values[0], entry_q['box_height'].values[0], entry_q['box_width'].values[0]]]), w=imgs_q[i].shape[0], h=imgs_q[i].shape[1])
        else:
            # TODO: Look into why our data needs this function...
            xyxy_q = xywh2xyxy_test(np.array([[entry_q['box_starty'].values[0], entry_q['box_startx'].values[0], entry_q['box_height'].values[0], entry_q['box_width'].values[0]]]))

        xyxy_qs.append(xyxy_q)

        # Normalise against width/height so we can compare if the images have different sizes between batches
        xyxy_n_q = normalise_xyxy(xyxy_q, imgs_q[i].shape[0], imgs_q[i].shape[1])
        xyxy_n_qs.append(xyxy_n_q)

        # Extract violation image for testing local features
        violation_imgs_q.append(imgs_q[i][int(xyxy_q[0][0]):int(xyxy_q[0][2]), int(xyxy_q[0][1]):int(xyxy_q[0][3])])
        
    # Loop for the radius and iou thresholding
    for i, fn_db in enumerate(fns_db):
        id_db = fn_db.split('/')[-1].split('.')[0]
        entry_db = violations_entry_db[violations_entry_db['id'] == int(id_db)]
        xyxy_n_db = xyxy_n_dbs[i]

        for j, fn_q in enumerate(fns_q):
            id_q = fn_q.split('/')[-1].split('.')[0]
            entry_q = violations_entry_q[violations_entry_q['id'] == int(id_q)]
            xyxy_n_q = xyxy_n_qs[i]

            # If we are using radius thresholding, set simularity of violations outside of radius to 0
            if radius_threshold is not None:
                distance = haversine(entry_db['lat'].to_list()[0], entry_db['long'].to_list()[0], entry_q['lat'].to_list()[0], entry_q['long'].to_list()[0])
                if(distance>radius_threshold):
                    S[i, j] = 0
                    continue

            if iou_threshold is not None:
                iou_score = iou(xyxy_n_db, xyxy_n_q)
                if iou_score < iou_threshold:
                    S[i, j] = 0
                    continue

    # Run through local feature comparison is using. Otherwise set S_local to 1s so it doesnt change results
    # TODO: logic isnt right, need to figure out best way to drop values based on the thresholding score of location first, as we can limit the scores we know arent close to one another
    # Then, we can apply the same thresholding procedure and combine? Need to think it through a bit
    
        
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
    
    if args.local_descriptor is not None:
        if args.local_descriptor == 'HDC-DELF':
            from feature_extraction.feature_extractor_holistic import HDCDELF
            local_feature_extractor = HDCDELF()

            print('===== Compute reference set descriptors for local violations')
            db_violation_holistic = feature_extractor.compute_features(violation_imgs_db)
            print('===== Compute query set descriptors for local violations')
            q_violation_holistic = feature_extractor.compute_features(violation_imgs_q)

            # normalize descriptors and compute S-matrix
            print('===== Compute cosine similarities S_local')
            db_violation_holistic = db_violation_holistic / np.linalg.norm(db_violation_holistic , axis=1, keepdims=True)
            q_violation_holistic = q_violation_holistic / np.linalg.norm(q_violation_holistic , axis=1, keepdims=True)
            S_local = np.matmul(db_violation_holistic , q_violation_holistic.transpose())

            # Limit based on thresholding matches (M2) from location matching
            for i, feat_db in enumerate(db_violation_holistic):
                for j, feat_q in enumerate(q_violation_holistic):
                    if M2[i, j] == False:
                        S_local[i, j] == 0

            print('test')
            print('===== Matching violations')
            # best match per query -> Single-best-match VPR
            M1_local = matching.best_match_per_query(S_local)

            # thresholding -> Multi-match VPR
            M2_local = matching.thresholding(S_local, 'auto')

            # Combining: use best match only when above threshold
            M3_local = np.logical_and(M1_local, M2_local)

            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(GThard)
            axs[0, 0].axis('off')
            axs[0, 0].set_title('GT')
            axs[0, 1].imshow(M1_local)
            axs[0, 1].axis('off')
            axs[0, 1].set_title('Best violation match per query')
            axs[1, 0].imshow(M2_local)
            axs[1, 0].axis('off')
            axs[1, 0].set_title('Thresholding S >= thresh')
            axs[1, 1].imshow(M3_local)
            axs[1, 1].axis('off')
            axs[1, 1].set_title('Best match >= thresh')
            fig.savefig(f'{output_dir}/M_local.png')
    else:
        M3_local = np.ones_like(S.shape)


    if args.local_descriptor_normalised is not None:
        if args.local_descriptor_normalised == 'HDC-DELF':
            from feature_extraction.feature_extractor_holistic import HDCDELF
            local_feature_extractor = HDCDELF()

            for i, img_db in enumerate(imgs_db):
                xyxy_db = xyxy_dbs[i]
                test1 = imgs_db[i][int(xyxy_db[0][0]):int(xyxy_db[0][2]), int(xyxy_db[0][1]):int(xyxy_db[0][3])]
                cv2.imwrite('test_before_db.png', test1)
                # xyxy_db_area = (xyxy_db[:,2] - xyxy_db[:,0])*(xyxy_db[:,3] - xyxy_db[:,1])
                for j, img_q in enumerate(imgs_q):
                    xyxy_q = xyxy_qs[j]
                    test2 = imgs_q[j][int(xyxy_db[0][0]):int(xyxy_q[0][2]), int(xyxy_q[0][1]):int(xyxy_q[0][3])]
                    cv2.imwrite('test_before_q.png', test2)

                    xyxy_db, xyxy_q = resize_bbox(xyxy_db, xyxy_q)
                    test1 = imgs_db[i][int(xyxy_db[0][0]):int(xyxy_db[0][2]), int(xyxy_db[0][1]):int(xyxy_db[0][3])]
                    cv2.imwrite('test_after_db.png', test1)

                    test2 = imgs_q[j][int(xyxy_q[0][0]):int(xyxy_q[0][2]), int(xyxy_q[0][1]):int(xyxy_q[0][3])]
                    cv2.imwrite('test_after_q.png', test2)


            print('===== Compute reference set descriptors for local violations')
            db_violation_holistic = feature_extractor.compute_features(violation_imgs_db)
            print('===== Compute query set descriptors for local violations')
            q_violation_holistic = feature_extractor.compute_features(violation_imgs_q)

            # normalize descriptors and compute S-matrix
            print('===== Compute cosine similarities S_local')
            db_violation_holistic = db_violation_holistic / np.linalg.norm(db_violation_holistic , axis=1, keepdims=True)
            q_violation_holistic = q_violation_holistic / np.linalg.norm(q_violation_holistic , axis=1, keepdims=True)
            S_local = np.matmul(db_violation_holistic , q_violation_holistic.transpose())

            # Limit based on thresholding matches (M2) from location matching
            for i, feat_db in enumerate(db_violation_holistic):
                for j, feat_q in enumerate(q_violation_holistic):
                    if M2[i, j] == False:
                        S_local[i, j] == 0

            print('test')
            print('===== Matching violations')
            # best match per query -> Single-best-match VPR
            M1_local = matching.best_match_per_query(S_local)

            # thresholding -> Multi-match VPR
            M2_local = matching.thresholding(S_local, 'auto')

            # Combining: use best match only when above threshold
            M3_local = np.logical_and(M1_local, M2_local)

            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(GThard)
            axs[0, 0].axis('off')
            axs[0, 0].set_title('GT')
            axs[0, 1].imshow(M1_local)
            axs[0, 1].axis('off')
            axs[0, 1].set_title('Best violation match per query')
            axs[1, 0].imshow(M2_local)
            axs[1, 0].axis('off')
            axs[1, 0].set_title('Thresholding S >= thresh')
            axs[1, 1].imshow(M3_local)
            axs[1, 1].axis('off')
            axs[1, 1].set_title('Best match >= thresh')
            fig.savefig(f'{output_dir}/M_local.png')
    else:
        M3_local = np.ones_like(S.shape)
    
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

def analyse_results():
    results_dir = glob.glob(f'results/HTT_example/*')
    overall_dataframe = pd.DataFrame()
    for i, result_dir in enumerate(results_dir):
        try:
            result = pd.read_csv(f'{result_dir}/results.csv')
            overall_dataframe = pd.concat([overall_dataframe, result], ignore_index=True)
        except FileNotFoundError as e:
            print(f'No results found in {result_dir}')
            continue
    
    print(f'Mean AUC: {np.mean(overall_dataframe['AUC'])}')
    print(f'Mean Precision: {np.mean(overall_dataframe['TP']/(overall_dataframe['TP']+overall_dataframe['FP']))}')
    print()
    print('stop')

if __name__ == "__main__":
    main()
    # analyse_results()
