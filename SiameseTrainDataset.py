from torch.utils.data import Dataset
import numpy as np
import torch

class SiameseTrainDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = np.array([label for _, label in dataset])
        self.classes = np.unique(self.labels)
        self.class_to_indices = {cls: np.where(self.labels == cls)[0]
                                 for cls in self.classes}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]

        same_class = np.random.choice([0, 1])
        if same_class:
            idx2 = np.random.choice(self.class_to_indices[label1])
        else:
            idx2 = np.random.choice(np.where(self.labels != label1)[0])
        img2, label2 = self.dataset[idx2]

        return img1, img2, torch.tensor(int(label1 != label2), dtype=torch.float32)
