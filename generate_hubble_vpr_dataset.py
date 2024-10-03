import pandas as pd
import os
import glob
import numpy as np
import shutil

input_dir = 'image_dataset'
output_dir = 'images/HTT_example'

datasets = os.listdir(input_dir)
for dataset in datasets:
    desired_ids = [1, 2, 5, 6, 20]
    scans_to_select = [(1, 2), (2, 3), (3, 4)]
    for scan_to_select in scans_to_select:
        for desired_id in desired_ids:
            try:
                match_dataset = pd.read_csv(f'{input_dir}/{dataset}/matches.csv')
            except Exception as e:
                continue
            match_dataset = match_dataset[match_dataset['rule_id'] == desired_id]
            if len(match_dataset) == 0:
                continue
            scan_1_name = match_dataset[f'scan_{scan_to_select[0]}_name'].to_list()[0]
            scan_2_name = match_dataset[f'scan_{scan_to_select[1]}_name'].to_list()[0]

            dataset_name = f'{desired_id}-{scan_1_name}-{scan_2_name}'
            os.makedirs(f'{output_dir}', exist_ok = True)
            if(os.path.exists(f'{output_dir}/{dataset_name}')):
                shutil.rmtree(f'{output_dir}/{dataset_name}')
            os.makedirs(f'{output_dir }/{dataset_name}/{scan_1_name}')
            os.makedirs(f'{output_dir }/{dataset_name}/{scan_2_name}')

            # Setting DB to be scan 1, and Q to be scan 2
            # Dropping all values of scan 1 that are nan (as we dont care at the moment if they arent there for now)
            match_dataset = match_dataset.dropna(subset=[f'scan_{scan_to_select[0]}_violation_id', f'scan_{scan_to_select[1]}_violation_id'])
            match_dataset = match_dataset.sort_values(by=[f'scan_{scan_to_select[0]}_violation_id'])
            match_dataset = match_dataset.reset_index()

            ids_db = np.array(sorted(match_dataset[f'scan_{scan_to_select[0]}_violation_id'].to_list()))
            ids_q = np.array(sorted(match_dataset[f'scan_{scan_to_select[1]}_violation_id'].to_list()))

            M_gt_hard = np.zeros([len(match_dataset), len(match_dataset)]).astype(bool)

            for i, row in match_dataset.iterrows():
                shutil.copy(f"../{row[f'scan_{scan_to_select[0]}_image_path']}", f"{output_dir}/{dataset_name}/{scan_1_name}/{int(row[f'scan_{scan_to_select[0]}_violation_id'])}.png")
                shutil.copy(f"../{row[f'scan_{scan_to_select[1]}_image_path']}", f"{output_dir}/{dataset_name}/{scan_2_name}/{int(row[f'scan_{scan_to_select[1]}_violation_id'])}.png")
                try:
                    j = np.where(np.array(ids_q) == row[f'scan_{scan_to_select[1]}_violation_id'])[0][0]
                except IndexError as e:
                    print('stop!')
                M_gt_hard[i, j] = True 
                
                # TODO: implement how we calculate soft GT? Maybe using locations
                # For now make a direct copy (might not be good)


            np.save(f'{output_dir}/{dataset_name}/M_gt_hard.npy', M_gt_hard)
            np.save(f'{output_dir}/{dataset_name}/M_gt_soft.npy', M_gt_hard)