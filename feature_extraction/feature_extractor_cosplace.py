
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data

from typing import List
import numpy as np
from tqdm.auto import tqdm


class ImageDataset(data.Dataset):
    def __init__(self, imgs):
        super().__init__()
        self.mytransform = self.input_transform()
        self.images = imgs

    def __getitem__(self, index):
        img = self.images[index]
        img = self.mytransform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    @staticmethod
    def input_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


class CosPlaceFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            print('Using GPU')
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print('Using MPS')
            self.device = torch.device("mps")
        else:
            print('Using CPU')
            self.device = torch.device("cpu")
        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                                    backbone="ResNet50", fc_output_dim=2048)
        self.dim = 2048
        self.model = self.model.to(self.device)

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        img_set = ImageDataset(imgs)
        test_data_loader = DataLoader(dataset=img_set, num_workers=4,
                                     batch_size=4, shuffle=False,
                                     pin_memory=torch.cuda.is_available())
        self.model.eval()
        with torch.no_grad():
            global_feats = np.empty((len(img_set), self.dim), dtype=np.float32)
            for input_data, indices in tqdm(test_data_loader):
                indices_np = indices.numpy()
                input_data = input_data.to(self.device)
                image_encoding = self.model(input_data)
                global_feats[indices_np, :] = image_encoding.cpu().numpy()
        return global_feats


