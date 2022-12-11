if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Program to train TinyYolo architecture on PascalVoc')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size (Default: 64)')
    parser.add_argument('-bn', '--use_batch_norm', default=False, type=bool, help='Use batch norm in Conv layers? (default: False)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial Learning rate (Default: 1e-3)')
    parser.add_argument('--max_epochs', default=100, type=int, help='Number of Epochs to train for (default: 100)')
    parser.add_argument('-p', '--print_freq', default=100, type=int, help='Log stats every `print_freq` iterations. (Default: 100)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay (Default: 5e-4)')
    parser.add_argument('--resume', default=None, type=str, help='Resume training from checkpoint')
    parser.add_argument('--workers', default=4, type=int, help='Number of Data loading workers (Default: 4)')



    args = parser.parse_args()

    import os
    import sys

    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from models import TinyNet
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms

    from data import PascalVocDataset

    from utils import train as train_epoch
    from utils import validate as validate_epoch
    from utils import adjust_learning_rate
    from utils import save_checkpoint

    from loss import DetectionLoss

    torch.autograd.set_detect_anomaly(True)

    train_transform = transforms.Compose([
        transforms.Resize((224*2, 224*2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224*2, 224*2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    
    train = PascalVocDataset(train=False, download=False, transform=train_transform)#PascalVocDataset(train=True, download=True)
    test = PascalVocDataset(train=False, download=False, transform=test_transform)


    train = DataLoader(train, batch_size=args.batch_size, shuffle=True)#num_workers=args.workers, pin_memory=True)
    test = DataLoader(test, batch_size=args.batch_size, shuffle=False)#num_workers=args.workers, pin_memory=True)

    net = TinyNet(use_batch_norm=args.use_batch_norm) #(is_feature_extractor=False, use_batch_norm=args.use_batch_norm).to('cuda') #dont even bother on cpu

    lossfn = DetectionLoss()

    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_iou = 0

    start_epoch = 0

    if args.resume != None:
        if os.path.isfile(args.resume):
            state = torch.load(args.resume)
            
            print(f'=> Resuming training from epoch {state["epoch"]} with best accuracy as {state["best_prec1"]}')

            best_prec1 = state['best_prec1']
            optimizer.load_state_dict(state['optimizer'])
            net.load_state_dict(state['state_dict'])

            start_epoch = state["epoch"] - 1

            print(f'=> Resumed Checkpoint Successfully')

        else:
            print('Could not resume training from checkpoint... File does not exist')
            sys.exit(1)


    for epoch in range(start_epoch, args.max_epochs):
        adjust_learning_rate(args.lr, optimzer=optimizer, epoch=epoch)

        train_epoch(
            train_loader=train,
            model=net,
            criterion=lossfn,
            optimizer=optimizer,
            epoch=epoch,
            print_freq=args.print_freq,
            metric='iou',
        )

        iou = validate_epoch(
            val_loader=test,
            model=net,
            criterion=lossfn,
            print_freq=args.print_freq,
            metric='iou'
        )


        best_iou = max(iou, iou)

        is_best =  iou > best_prec1

        if is_best:

            save_checkpoint(
                state={
                    'epoch': epoch+1,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': net.state_dict(),
                    'best_iou': best_iou,
                },
                dir='checkpoints/tinyyolo/best',
                is_best=is_best,
            )
        else:
            save_checkpoint(
                state={
                    'epoch': epoch+1,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': net.state_dict(),
                    'best_iou': best_iou,
                },
                dir='checkpoints/tinyyolo',
                is_best=is_best,
            )