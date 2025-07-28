# main.py
import os
from config import parse_option
from data_preprocessing import (
    set_loader, set_model, set_optimizer,
    save_model, train_supervised, sample_evaluation
)

args = [
    '--batch_size', '64',
    '--model', 'resnet101',
    '--dataset', 'OLIVES',
    '--epochs', '30',
    '--device', 'cuda:0',
    '--train_image_path', 'E:/Backup/My backup/Project Files/Competitions/VIPCup/kaggle/input/olives-vip-cup-2023/2023 IEEE SPS Video and Image Processing (VIP) Cup - Ophthalmic Biomarker Detection/TRAIN/OLIVES',
    '--test_image_path', 'E:/Backup/My backup/Project Files/Competitions/VIPCup/kaggle/input/olives-vip-cup-2023/2023 IEEE SPS Video and Image Processing (VIP) Cup - Ophthalmic Biomarker Detection/TEST/',
    '--test_csv_path', 'E:/Backup/My backup/Project Files/Competitions/VIPCup/kaggle/input/olives-vip-cup-2023/2023 IEEE SPS Video and Image Processing (VIP) Cup - Ophthalmic Biomarker Detection/TEST/test_set_submission_template.csv',
    '--train_csv_path', 'E:/Backup/My backup/Project Files/Competitions/VIPCup/kaggle/input/olives-training-labels/Training_Biomarker_Data.csv'
]
opt = parse_option(args)

train_loader, val_loader, test_loader = set_loader(opt)
model, criterion = set_model(opt)
optimizer = set_optimizer(opt, model)

for epoch in range(1, opt.epochs + 1):
    train_supervised(train_loader, model, criterion, optimizer, epoch, opt)

save_file = os.path.join(opt.save_folder, 'last.pth')
save_model(model, optimizer, opt, opt.epochs, save_file)

sample_evaluation(val_loader, model, opt)
