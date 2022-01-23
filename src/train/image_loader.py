from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # super().__init__(Dataset)
        self.csv_file = csv_file
        self.transform = transform
        df = pd.read_csv(self.csv_file)
        self.images = df['image']
        self.labels = df['label']
        self.classes = self.num_classes()

    def one_hot_encode(self, label):
        one_hot = np.zeros(self.classes)
        one_hot[label] = 1
        return one_hot

    def __getitem__(self, item):
        image = self.images[item]
        label = int(self.labels[item])
        # label = self.one_hot_encode(label)
        img = Image.open(image)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def num_classes(self):
        return len(set(self.labels))
