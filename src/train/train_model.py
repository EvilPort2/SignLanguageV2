from torchvision.transforms import transforms
import torch.nn as nn
from image_loader import ImageDataset
from torch.utils.data import DataLoader
import time
import copy
import numpy as np
import torch
from tqdm import tqdm
from model import create_model, set_parameter_requires_grad
import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser('Sign Language model trainer')
parser.add_argument('--model_arch', help='Model architecture to train')
parser.add_argument('--batch_size',  help='Batch Size', default=100, type=int)
parser.add_argument('--lr',  help='Learning rate', default=0.00001, type=float)
parser.add_argument('--train_csv', help='Path to train.csv')
parser.add_argument('--val_csv', help='Path to validation.csv')
parser.add_argument('--test_csv', help='Path to test.csv')
parser.add_argument('--model_save_dir', help='Save model to this directory')

args = parser.parse_args()

model_save_dir = args.model_save_dir
train_csv = args.train_csv
val_csv = args.val_csv
test_csv = args.test_csv
model_arch = args.model_arch
batch_size = args.batch_size
lr = args.lr

Path(model_save_dir).mkdir(exist_ok=True, parents=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloader(csv_file, batch_size=50):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset = ImageDataset(csv_file, data_transforms)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, num_workers=0)
    return image_dataloader, image_dataset.classes


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'model.pt'))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train():
    criterion = nn.CrossEntropyLoss().to(device)
    train_dataloader, num_classes = create_dataloader(train_csv, batch_size=batch_size)
    val_dataloader, _ = create_dataloader(val_csv, batch_size=batch_size)
    test_dataloader, _ = create_dataloader(test_csv, batch_size=batch_size)
    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }

    model = create_model(model_arch, num_classes)
    model.to(device)
    labels = {
        'labels': torch.Tensor(np.arange(num_classes))
    }
    torch.save(labels, os.path.join(model_save_dir, 'labels.pt'))
    set_parameter_requires_grad(model, feature_extracting=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model, _ = train_model(model, dataloaders, criterion, optimizer, num_epochs=15)

    torch.save(model.state_dict(), os.path.join(model_save_dir, 'model.pt'))


if __name__ == '__main__':
    train()

