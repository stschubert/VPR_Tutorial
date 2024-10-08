import pandas as pd
import os
import glob
import numpy as np
import shutil
import copy
input_dir = 'image_dataset'
output_dir = 'images/HTT_example'

datasets = os.listdir(input_dir)

for dataset in datasets:

    # Desired violation IDs. Filtering here means we reduce false positives when we use VPR
    desired_ids = [1, 2, 5, 6, 20]

    # Dataset not saved in order (ie time(scan_1) is not always <= time(scan_2)... so we sort through names (YYMM) to ensure we match between batches by time)
    # This is how we did the analysis of HTT previously, so recreating it here
    # Currently only move between violations temporally for now, add in logic for tracking new violations later
    scans_to_select = [(0, 1), (1, 2), (2, 3)]

    scan_dirs = sorted([ name for name in os.listdir(f'{input_dir}/{dataset}') if os.path.isdir(os.path.join(f'{input_dir}/{dataset}', name)) ])
    try:
        match_dataset_original = pd.read_csv(f'{input_dir}/{dataset}/matches.csv')
    except Exception as e:
        continue

    # Loop through scan pairs
    for scan_to_select in scans_to_select:
        
        #Need to get correct index of scan name
        scan_1_name = None
        scan_2_name = None

        # Find the correct index in the dataset and save for reference later
        for _, val in enumerate([1, 2, 3, 4]):
            try:
                if np.any([scan_dirs[scan_to_select[0]] in x for x in match_dataset_original[f'scan_{val}_name'].to_list()]):
                    scan_1_index = val
                    scan_1_name = match_dataset_original[f'scan_{val}_name'].to_list()[0]
                if np.any([scan_dirs[scan_to_select[1]] in x for x in match_dataset_original[f'scan_{val}_name'].to_list()]):
                    scan_2_index = val
                    scan_2_name = match_dataset_original[f'scan_{val}_name'].to_list()[0]
            except IndexError as e:
                continue
        
        for desired_id in desired_ids:
            
            # Drop violations that we are not interested in
            match_dataset = copy.copy(match_dataset_original)
            match_dataset = match_dataset[match_dataset['rule_id'] == desired_id]
            if len(match_dataset) == 0:
                continue
            
            # Make output dirs. Delete if they already exist
            dataset_name = f'{desired_id}-{scan_1_name}-{scan_2_name}'
            
            os.makedirs(f'{output_dir}', exist_ok = True)
            if(os.path.exists(f'{output_dir}/{dataset_name}')):
                shutil.rmtree(f'{output_dir}/{dataset_name}')
            os.makedirs(f'{output_dir }/{dataset_name}/{scan_1_name}')
            os.makedirs(f'{output_dir }/{dataset_name}/{scan_2_name}')

            # Setting DB (reference) to be scan 1, and Q (query) to be scan 2
            # Dropping all values of scan 1 that are nan, keeping any values in scan 2 that are, and also removing any erroneous double violation mappings
            match_dataset = match_dataset.dropna(subset=[f'scan_{scan_1_index}_violation_id', f'scan_{scan_2_index}_violation_id'])  
            # match_dataset = match_dataset[(~match_dataset.duplicated([f'scan_{scan_1_index}_violation_id', f'scan_{scan_2_index}_violation_id'])) | (match_dataset[f'scan_{scan_2_index}_violation_id'].isnull())]
            match_dataset = match_dataset[(~match_dataset.duplicated([f'scan_{scan_1_index}_violation_id', f'scan_{scan_2_index}_violation_id']))]
            match_dataset = match_dataset.sort_values(by=[f'scan_{scan_1_index}_violation_id'])
            match_dataset = match_dataset.reset_index()

            ids_db = np.array(sorted(match_dataset[f'scan_{scan_1_index}_violation_id'].to_list()))
            ids_q = np.array(sorted(match_dataset[f'scan_{scan_2_index}_violation_id'].to_list()))

            image_order = {}
            image_order[scan_1_name] = match_dataset[f'scan_{scan_1_index}_violation_id'].to_list()
            image_order[scan_2_name] = match_dataset[f'scan_{scan_2_index}_violation_id'].to_list()

            # Creating empty ground truth array
            M_gt_hard = np.zeros([len(image_order[scan_1_name]), len(image_order[scan_2_name])]).astype(bool)

            # Loop through dataset matches, copy images into the correct directory and set the [i, j] of the matrix to true to signify a TP match
            for i, row in match_dataset.iterrows():
                try:
                    shutil.copy(f"{row[f'scan_{scan_1_index}_image_path']}", f"{output_dir}/{dataset_name}/{scan_1_name}/{int(row[f'scan_{scan_1_index}_violation_id'])}.png")
                except (FileNotFoundError, ValueError) as e:
                    continue

                try:
                    shutil.copy(f"{row[f'scan_{scan_2_index}_image_path']}", f"{output_dir}/{dataset_name}/{scan_2_name}/{int(row[f'scan_{scan_2_index}_violation_id'])}.png")
                except (FileNotFoundError, ValueError) as e:
                    continue

                try:
                    j = np.where(np.array(ids_q) == row[f'scan_{scan_2_index}_violation_id'])[0][0]
                except IndexError as e:
                    continue

                M_gt_hard[i, j] = True 
                
            # TODO: implement how we calculate soft GT? Maybe using locations
            # For now make a direct copy (might make FP measurements a little harsh in original VPR code

            # Save matrices into output directories
            np.save(f'{output_dir}/{dataset_name}/M_gt_hard.npy', M_gt_hard)
            np.save(f'{output_dir}/{dataset_name}/M_gt_soft.npy', M_gt_hard)
            np.save(f'{output_dir}/{dataset_name}/image_order.npy', image_order)
            # Save dataset info in each one
            try:
                shutil.copy(f'{input_dir}/{dataset}/{scan_1_name}.csv', f"{output_dir}/{dataset_name}/{scan_1_name}.csv")
                shutil.copy(f'{input_dir}/{dataset}/{scan_2_name}.csv', f"{output_dir}/{dataset_name}/{scan_2_name}.csv")
            except Exception as e:
                continue