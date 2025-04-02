from torch.utils.data import Dataset
import numpy as np

class SiameseTestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = np.array([label for _, label in dataset])
        self.classes = np.unique(self.labels)
        self.class_to_indices = {cls: np.where(self.labels == cls)[0]
                                 for cls in self.classes}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def sample_episode(self, n_way, k_shot, num_queries):
        selected_classes = np.random.choice(self.classes, n_way, replace=False)
        support = []
        query = []

        for cls in selected_classes:
            indices = np.random.choice(
                self.class_to_indices[cls],
                k_shot + num_queries,
                replace=False
            )
            support.extend(indices[:k_shot])
            query.extend(indices[k_shot:])

        return support, query, selected_classes