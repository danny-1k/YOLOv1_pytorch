import os
import shutil

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Tuple

import time

#https://github.com/pytorch/examples/tree/master/imagenet
#https://github.com/wenwei202/pytorch-examples/blob/master/imagenet/main.py
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor,
             target: torch.Tensor,
             topk: Tuple[int] = (1)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res


def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def train(train_loader: DataLoader,
          model: nn.Module,
          criterion: nn.Module,
          optimizer: Optimizer,
          epoch: int,
          print_freq: int = 100
         ):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training
        input = input.cuda(non_blocking=True)
        target = input.cuda(non_blocking=True)

        # compute output

        logits = model(input)
        loss = criterion(logits, target)

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        if i % print_freq == 0:
            # Every log_freq iterations, check the loss, accuracy and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<-> device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(logits.data, target.data, topk=(1,5))


            # to_python_float incurs a host<->device sync
            batch_size = input[0].size(0)
            losses.update(to_python_float(loss), batch_size)
            top1.update(to_python_float(prec1), batch_size)
            top5.update(to_python_float(prec5), batch_size)

            batch_time.update((time.time() - end) / print_freq)
            end = time.time()

            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Speed {batch_size / batch_time.val:.3f} ({batch_size / batch_time.avg:.3f})\t"
                f"Loss {losses.val:.10f} ({losses.avg:.4f})\t"
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Prec@5 {top5.val:.3f} ({top5.avg:3f})"
            )

def adjust_learning_rate(initial_lr: float,
                         optimzer: Optimizer,
                         epoch: int
                        ):
    """Sets the learning rate to the initial LR decayed by 10 ever 30 epochs"""
    lr = initial_lr * (.1 ** (epoch // 30))
    for param_group in optimzer.param_groups:
        param_group['lr'] = lr


def validate(
            val_loader: DataLoader,
            model: nn.Module,
            criterion: nn.Module, 
            print_freq: int = 100
            ):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state: dict,
                    dir: str,
                    is_best: bool
                ):

    if not os.path.exists(dir):
        os.makedirs(dir)

    torch.save(state, os.path.join(dir, 'checkpoint.pth.tar'))

    if is_best:
        shutil.copy(os.path.join(dir, 'checkpoint.pth.tar'), os.path.join(dir, 'best.pth.tar'))