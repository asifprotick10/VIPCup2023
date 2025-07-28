# data_preprocessing.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import sys
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.metrics import f1_score
from model import ResNet
from datasets import OLIVES, RECOVERY

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_model(opt):
    device = opt.device
    model = ResNet(name=opt.model, num_classes=opt.ncls)
    criterion = nn.BCEWithLogitsLoss()
    return model.to(device), criterion.to(device)

def set_loader(opt):
    mean, std = (.1706,), (.2112,)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    csv_path_train = opt.train_csv_path
    csv_path_test = opt.test_csv_path
    data_path_train = opt.train_image_path
    data_path_test = opt.test_image_path
    train_dataset = OLIVES(csv_path_train, data_path_train, transforms=train_transform)
    test_dataset = RECOVERY(csv_path_test, data_path_test, transforms=val_transform)
    train_dataset, val_dataset = random_split(train_dataset, [0.95, 0.05], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader

def set_optimizer(opt, model):
    return optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def train_supervised(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    correct_predictions = 0
    device = opt.device
    end = time.time()

    for idx, (image, bio_tensor) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = image.to(device)
        labels = bio_tensor.float().to(device)
        bsz = labels.shape[0]

        output = model(images)
        loss = criterion(output, labels)
        predicted_labels = torch.round(torch.sigmoid(output))
        correct_predictions += (predicted_labels == labels).sum().item()
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader)))
            sys.stdout.flush()

    total_values = len(train_loader.dataset) * 6
    training_accuracy = (correct_predictions / total_values) * 100.0
    print(f"Training Accuracy: {training_accuracy:.2f}%")
    return losses.avg

def sample_evaluation(val_loader, model, opt):
    model.eval()
    device = opt.device
    out_list, label_list = [], []
    correct_count, total_count = 0, 0

    with torch.no_grad():
        for idx, (image, bio_tensor) in enumerate(val_loader):
            images = image.float().to(device)
            labels = bio_tensor.float()
            label_list.append(labels.squeeze().cpu().numpy())
            output = model(images)
            output = torch.round(torch.sigmoid(output))
            out_list.append(output.squeeze().cpu().numpy())
            correct_count += (labels.to(device) == output.to(device)).sum()
            total_count += len(labels)

    print((correct_count / total_count) * 100, "%")
    f = f1_score(np.array(label_list), np.array(out_list), average='macro')
    print(f)
