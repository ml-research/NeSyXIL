from __future__ import print_function
import argparse
import uuid
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import data as data
import utils as utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
	        nn.Flatten(1),
            nn.Linear(12544, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
        self.dropout1 = nn.Dropout(0.25)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    train_loss /= len(train_loader.dataset)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Acc/train', 100. * correct / len(train_loader.dataset), epoch)


def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Acc/test', 100. * correct / len(test_loader.dataset), epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-dir', type=str, default='../data/data/',
                        help='directory to data')
    parser.add_argument('--fp-ckpt', type=str, default='None',
                        help='directory to model ckpt')
    parser.add_argument('--mode', type=str, default='train',
                        help='trian or plot')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    trainloader, validloader, testloader, nb_classes, dim_inp = data.get_dataset(args.data_dir, 'cmnist', args.batch_size)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if args.mode == 'train':

        run_id = 'default_' + str(uuid.uuid1())
        writer = SummaryWriter(log_dir='runs/' + run_id, comment="_" + "_id_{}".format(run_id))

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, trainloader, optimizer, epoch, writer)
            test(model, device, testloader, epoch, writer)
        #     scheduler.step()

        if args.save_model:
            results = {
                "weights": model.state_dict(),
                "args": args,
            }
            torch.save(results, f"runs/{run_id}/mnist_cnn.pt")

        save_dir = writer.log_dir + '/figures/'
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            # directory already exists
            pass
        create_plots(model, testloader, save_dir)

    elif args.mode == 'plot':
        assert args.fp_ckpt

        model = Net().to(device)
        ckpt = torch.load(args.fp_ckpt)
        model.load_state_dict(ckpt['weights'])
        model.eval()

        save_dir = args.fp_ckpt.split('mnist_cnn')[0] + 'figures/'
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            # directory already exists
            pass

        # utils.save_expl_images(model, testloader, 'test', save_dir)

        utils.create_overview_plots(trainloader, testloader, save_dir)


if __name__ == '__main__':
    main()