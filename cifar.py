# MIT License

# Copyright (c) 2022 Chengfeng Zhou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.==============================================================================
"""Main script to launch WIN-WIN training on CIFAR-10/100.

Supports ResNet18 models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `  python cifar.py \
    --dataset cifar10 \
    --epochs 180 \
    --lr 0.3 \
    --batch-size 64 \
    --WIN-WIN online \
  `
"""
from __future__ import print_function

import argparse
import os
import shutil
import datetime
import time

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models


from WIN import WindowNorm2d

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--data-path',
    type=str,
    default='./data',
    required=True,
    help='Path to CIFAR and CIFAR-C directories')

parser.add_argument(
    '--WIN-WIN',
    type=str,
    default='disabled',
    choices=['disabled', 'offline', 'online'],
    help='Apply the WIN-WIN.')

# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=180, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.3,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=64)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

args = parser.parse_args()
print(args)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout",
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur",
    "lines", "sparkles", "transverse_chromatic_abberation"]

NUM_CLASSES = 100 if args.dataset == 'cifar100' else 10


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))


def train(net, train_loader, optimizer, scheduler, args, scaler):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    total_correct = 0.
    total = 0.
    for i, (images, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        images = images.cuda()
        targets = targets.cuda()

        with autocast():
            if args.WIN_WIN != 'disabled':
                net.eval()

                logits_eval = net(images)

                net.train()

                logits = net(images)

                p_loss = F.kl_div(F.log_softmax(logits, dim=-1),
                                  F.softmax(logits_eval, dim=-1), reduction='none')
                q_loss = F.kl_div(F.log_softmax(
                    logits_eval, dim=-1), F.softmax(logits, dim=-1), reduction='none')

                jsd_loss = (p_loss.sum() + q_loss.sum()) / 2

                loss = 0.5 * (F.cross_entropy(logits_eval, targets) +
                              F.cross_entropy(logits, targets)) + 0.3 * jsd_loss
            else:
                logits = net(images)
                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_ema = loss_ema * 0.9 + float(loss) * 0.1

            pred = logits.data.max(1)[1]

            total_correct += pred.eq(targets.data).sum().item()
            total += targets.shape[0]

            # if i % args.print_freq == 0:
            #   print('Train Loss {:.3f}'.format(loss_ema))

    return loss_ema, total_correct / total


def test(net, test_loader, adv=None):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            # adversarial
            if adv:
                images = adv(net, images, targets)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            # print(loss)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader), total_correct / len(
        test_loader.dataset)


def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    corrs = CBAR_CORRUPTIONS if 'Bar' in base_path else CORRUPTIONS
    for corruption in corrs:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loss, test_acc = test(net, test_loader)
        corruption_accs.append(test_acc)
        print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    return np.mean(corruption_accs)


def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm


class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        # unnormalize
        bx = (bx+1)/2

        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon,
                                                    self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(
                torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx*2-1


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    # Load datasets
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), normalize])

    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            os.path.join(args.data_path, 'cifar'), train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            os.path.join(args.data_path, 'cifar'), train=False, transform=test_transform, download=True)
        base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C/')
        base_c_bar_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C-Bar/')
        num_classes = 10
    else:
        train_data = datasets.CIFAR100(
            os.path.join(args.data_path, 'cifar'), train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            os.path.join(args.data_path, 'cifar'), train=False, transform=test_transform, download=True)
        base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-100-C/')
        base_c_bar_path = os.path.join(
            args.data_path, 'cifar/CIFAR-100-C-Bar/')
        num_classes = 100

    # Fix dataloader worker issue
    # https://github.com/pytorch/pytorch/issues/5059
    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=wif)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    # Create model

    net = models.resnet18(pretrained=False, num_classes=num_classes)

    if args.WIN_WIN == 'online':
        net = WindowNorm2d.convert_WIN_model(net)
    elif args.WIN_WIN == 'offline':
        net = WindowNorm2d.convert_WIN_model(net, cached=True)

    print(net)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    scaler = GradScaler()

    # initialize adversary
    adversary = PGD(epsilon=2./255, num_steps=1, step_size=2./255).cuda()

    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model restored from epoch:', start_epoch)

    if args.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc = test(net, test_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        adv_test_loss, adv_test_acc = test(net, test_loader, adv=adversary)
        print('Adversarial\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            adv_test_loss, 100 - 100. * adv_test_acc))

        test_c_acc = test_c(net, test_data, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
        return

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    time_now = datetime.datetime.now()
    id = '{}-{}-{}'.format(str(time_now.date()),
                           time_now.hour, time_now.minute)

    if args.WIN_WIN != 'disabled':
        args.save = 'resnet18_WIN_' + id
    else:
        args.save = 'resnet18_BN_' + id

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    elif args.save != 'snapshots':
        raise Exception('%s exists' % args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    log_path = os.path.join(args.save,
                            args.dataset + '_resnet18_training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,train_acc(%),test_acc(%)\n')

    best_acc = 0
    print('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, args.epochs):
        begin_time = time.time()

        train_loss_ema, train_acc = train(
            net, train_loader, optimizer, scheduler, args, scaler)
        end_time = time.time()
        test_loss, test_acc = test(net, test_loader)
        # test_c_acc = test_c(net, test_data, base_c_path)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            'epoch': epoch,
            'dataset': args.dataset,
            'model': 'resnet18',
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(args.save, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(
                args.save, 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f,%0.2f\n' % (
                (epoch + 1),
                end_time - begin_time,
                train_loss_ema,
                test_loss,
                train_acc,
                test_acc,
                # 100 - 100. * test_c_acc,
            ))

        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Train Acc {4:.2f} % | Test Acc {4:.2f} %'
            .format((epoch + 1), int(end_time - begin_time), train_loss_ema, test_loss, 100.*train_acc, 100.*test_acc))

    _, adv_test_acc = test(net, test_loader, adv=adversary)
    print('Adversarial Test Error: {:.3f}\n'.format(100 - 100. * adv_test_acc))

    test_c_acc = test_c(net, test_data, base_c_path)
    print('Mean C Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_acc))

    # test_c_bar_acc = test_c(net, test_data, base_c_bar_path)
    # print('Mean C-Bar Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_bar_acc))

    # print('Mean Corruption Error: {:.3f}\n'.format(100 - 100. * (15*test_c_acc + 10*test_c_bar_acc)/25))

    # with open(log_path, 'a') as f:
    #   f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
    #           (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
    main()