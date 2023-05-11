import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import subprocess

from PIL import Image

from os.path import join, isfile
from typing import List
import numpy as np
from tqdm.auto import tqdm

from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

from .feature_extractor import FeatureExtractor


class ImageDataset(data.Dataset):
    def __init__(self, imgs):
        super().__init__()
        self.mytransform = self.input_transform()
        self.images = imgs

    def __getitem__(self, index):
        # img = Image.open(self.images[index])
        # TODO: Check if the channel order is correct
        img = self.images[index]
        img = self.mytransform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    @staticmethod
    def input_transform(resize=(480, 640)):
        if resize[0] > 0 and resize[1] > 0:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

class PatchNetVLADFeatureExtractor(FeatureExtractor):
    def __init__(self, config):
        self.config = config

        if torch.cuda.is_available():
            print('Using GPU')
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print('Using MPS')
            self.device = torch.device("mps")
        else:
            print('Using CPU')
            self.device = torch.device("cpu")

        encoder_dim, encoder = get_backend()

        if self.config['global_params']['num_pcs'] != '0':
            resume_ckpt = self.config['global_params']['resumePath'] + self.config['global_params']['num_pcs'] + '.pth.tar'
        else:
            resume_ckpt = self.config['global_params']['resumePath'] + '.pth.tar'

        if not isfile(resume_ckpt):
            resume_ckpt = join(PATCHNETVLAD_ROOT_DIR, resume_ckpt)
            if not isfile(resume_ckpt):
                print('Downloading Patch-NetVLAD models, this might take a while ...')
                subprocess.run(["patchnetvlad-download-models"])


        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            if self.config['global_params']['num_pcs'] != '0':
                assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(self.config['global_params']['num_pcs'])
            self.config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

            if self.config['global_params']['num_pcs'] != '0':
                use_pca = True
            else:
                use_pca = False
            self.model = get_model(encoder, encoder_dim, self.config['global_params'], append_pca_layer=use_pca)
            self.model.load_state_dict(checkpoint['state_dict'])

            if int(self.config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
                self.model.encoder = torch.nn.DataParallel(self.model.encoder)
                self.model.pool = torch.nn.DataParallel(self.model.pool)

            self.model = self.model.to(self.device)
            print(f"=> loaded checkpoint '{resume_ckpt}'")
        else:
            raise FileNotFoundError(f"=> no checkpoint found at '{resume_ckpt}'")
        
        if self.config['global_params']['pooling'].lower() == 'patchnetvlad':
            self.num_patches = self.get_num_patches()
        else:
            self.num_patches = None


    def get_num_patches(self):
        H = int(int(self.config['feature_match']['imageresizeH']) / 16)  # 16 is the vgg scaling from image space to feature space (conv5)
        W = int(int(self.config['feature_match']['imageresizeW']) / 16)
        padding_size = [0, 0]
        patch_sizes = [int(s) for s in self.config['global_params']['patch_sizes'].split(",")]
        patch_size = (int(patch_sizes[0]), int(patch_sizes[0]))
        strides = [int(s) for s in self.config['global_params']['strides'].split(",")]
        stride = (int(strides[0]), int(strides[0]))

        Hout = int((H + (2 * padding_size[0]) - patch_size[0]) / stride[0] + 1)
        Wout = int((W + (2 * padding_size[1]) - patch_size[1]) / stride[1] + 1)

        num_regions = Hout * Wout
        return num_regions


    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        pool_size = int(self.config['global_params']['num_pcs'])

        img_set = ImageDataset(imgs)
        test_data_loader = DataLoader(dataset=img_set, num_workers=int(self.config['global_params']['threads']),
                                    batch_size=int(self.config['feature_extract']['cacheBatchSize']),
                                    shuffle=False, pin_memory=torch.cuda.is_available())

        self.model.eval()
        with torch.no_grad():
            global_feats = np.empty((len(img_set), pool_size), dtype=np.float32)
            if self.config['global_params']['pooling'].lower() == 'patchnetvlad':
                patch_feats = np.empty((len(img_set), pool_size, self.num_patches), dtype=np.float32)

            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader), 1):
                indices_np = indices.detach().numpy()
                input_data = input_data.to(self.device)
                image_encoding = self.model.encoder(input_data)

                if self.config['global_params']['pooling'].lower() == 'patchnetvlad':
                    vlad_local, vlad_global = self.model.pool(image_encoding)

                    vlad_global_pca = get_pca_encoding(self.model, vlad_global)
                    global_feats[indices_np, :] = vlad_global_pca.detach().cpu().numpy()

                    for this_local in vlad_local:
                        patch_feats_batch = np.empty((this_local.size(0), pool_size, this_local.size(2)),
                                                dtype=np.float32)
                        grid = np.indices((1, this_local.size(0)))
                        this_local_pca = get_pca_encoding(self.model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))).\
                            reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
                        patch_feats_batch[grid, :, :] = this_local_pca.detach().cpu().numpy()
                        for i, val in enumerate(indices_np):
                            patch_feats[val] = patch_feats_batch[i]
                else:
                    vlad_global = self.model.pool(image_encoding)
                    vlad_global_pca = get_pca_encoding(self.model, vlad_global)
                    global_feats[indices_np, :] = vlad_global_pca.detach().cpu().numpy()

        if self.config['global_params']['pooling'].lower() == 'patchnetvlad':
            return global_feats, patch_feats
        else:
            return global_feats


    def local_matcher_from_numpy_single_scale(self, input_query_local_features_prefix, input_index_local_features_prefix):
        from patchnetvlad.models.local_matcher import normalise_func, calc_keypoint_centers_from_patches
        from patchnetvlad.tools.patch_matcher import PatchMatcher

        patch_sizes = [int(s) for s in self.config['global_params']['patch_sizes'].split(",")]
        assert(len(patch_sizes) == 1)
        strides = [int(s) for s in self.config['global_params']['strides'].split(",")]
        patch_weights = np.array(self.config['feature_match']['patchWeights2Use'].split(",")).astype(float)

        all_keypoints = []
        all_indices = []

        for patch_size, stride in zip(patch_sizes, strides):
            # we currently only provide support for square patches, but this can be easily modified for future works
            keypoints, indices = calc_keypoint_centers_from_patches(self.config['feature_match'], patch_size, patch_size, stride, stride)
            all_keypoints.append(keypoints)
            all_indices.append(indices)

        raw_diffs = []

        matcher = PatchMatcher(self.config['feature_match']['matcher'], patch_sizes, strides, all_keypoints, all_indices)

        for q_idx in tqdm(range(len(input_query_local_features_prefix)), leave=False, desc='Patch compare pred'):
            diffs = np.zeros((len(input_index_local_features_prefix), len(patch_sizes)))
            qfeat = [torch.transpose(torch.tensor(input_query_local_features_prefix[q_idx], device=self.device), 0, 1)]

            for candidate in range(len(input_index_local_features_prefix)):
                dbfeat = [torch.tensor(input_index_local_features_prefix[candidate], device=self.device)]
                diffs[candidate, :], _, _ = matcher.match(qfeat, dbfeat)

            diffs = normalise_func(diffs, len(patch_sizes), patch_weights)
            raw_diffs.append(diffs)

        return np.array(raw_diffs).T * -1
