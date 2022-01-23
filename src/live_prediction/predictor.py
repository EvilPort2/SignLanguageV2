import torch
import sys
sys.path.append('..')
from models.model import create_model
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        labels_file = os.path.join(self.model_dir, 'label2string.json')
        self.labels = json.loads(open(labels_file).read())
        self.num_classes = len(self.labels)
        self.model_path = os.path.join(self.model_dir, 'model.pt')
        state_dict = torch.load(self.model_path)
        self.model = create_model(self.num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        del state_dict

    def __call__(self, img):
        if type(img) == np.array:
            img = Image.fromarray(img)
        img = self.data_transforms(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)
        output = self.model(img)
        label = torch.max(output, 1)[1].detach().to('cpu').numpy()[0]
        label = self.labels[str(label)]
        return label


if __name__ == '__main__':
    predictor = Predictor(os.path.join('D:\\', 'SignLanguageV2', 'models'))
    img = Image.open('D:/SignLanguageV2/data/green_screen/5/55.png')
    print(predictor(img))