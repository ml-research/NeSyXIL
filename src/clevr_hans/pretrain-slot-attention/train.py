import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp

import scipy.optimize
import numpy as np
from tqdm import tqdm
import matplotlib
from torch.optim import lr_scheduler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import data
import model
import utils as utils

torch.set_num_threads(6)

def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--ap-log", type=int, default=10, help="Number of epochs before logging AP"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["clevr-state"],
        help="Use MNIST dataset",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")

    parser.add_argument("--data-dir", type=str, help="Directory to data")
    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')

    args = parser.parse_args()
    return args


def run(net, loader, optimizer, criterion, writer, args, train=False, epoch=0, pool=None):
    if train:
        net.train()
        prefix = "train"
        torch.set_grad_enabled(True)
    else:
        net.eval()
        prefix = "test"
        torch.set_grad_enabled(False)

        preds_all = torch.zeros(0, args.n_slots, args.n_attr)
        target_all = torch.zeros(0, args.n_slots, args.n_attr)

    iters_per_epoch = len(loader)

    for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        # input is either a set or an image
        imgs, target_set = map(lambda x: x.cuda(), sample)

        output = net.forward(imgs)

        loss = utils.hungarian_loss(output, target_set, thread_pool=pool)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("metric/train_loss", loss.item(), global_step=i)
            print(f"Epoch {epoch} Train Loss: {loss.item()}")

        else:
            if i % iters_per_epoch == 0:
                # print predictions for one image, match predictions with targets
                matched_output = utils.hungarian_matching(target_set[:2], output[:2].to('cuda'), verbose=0)
                # for k in range(2):
                print(f"\nGT: \n{np.round(target_set.detach().cpu().numpy()[0, 0], 2)}")
                print(f"\nPred: \n{np.round(matched_output.detach().cpu().numpy()[0, 0], 2)}\n")

                preds_all = torch.cat((preds_all, output), 0)
                target_all = torch.cat((target_all, target_set), 0)

            writer.add_scalar("metric/val_loss", loss.item(), global_step=i)


def main():
    args = get_args()

    writer = SummaryWriter(f"runs/{args.name}", purge_step=0)
    # writer = None
    utils.save_args(args, writer)

    dataset_train = data.CLEVR(
        args.data_dir, "train",
    )
    dataset_test = data.CLEVR(
        args.data_dir, "val",
    )

    if not args.eval_only:
        train_loader = data.get_loader(
            dataset_train, batch_size=args.batch_size, num_workers=args.num_workers
        )
    if not args.train_only:
        test_loader = data.get_loader(
            dataset_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    net = model.SlotAttention_model(n_slots=10, n_iters=3, n_attr=18,
                                    encoder_hidden_channels=64,
                                    attention_hidden_channels=128)
    args.n_attr = net.n_attr

    start_epoch = 0
    if args.resume:
        print("Loading ckpt ...")
        log = torch.load(args.resume)
        weights = log["weights"]
        net.load_state_dict(weights, strict=True)
        start_epoch = log["args"]["epochs"]


    if not args.no_cuda:
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00005)
    criterion = torch.nn.SmoothL1Loss()

    # store args as txt file
    utils.save_args(args, writer)

    for epoch in np.arange(start_epoch, args.epochs + start_epoch):
        with mp.Pool(10) as pool:
            if not args.eval_only:
                run(net, train_loader, optimizer, criterion, writer, args, train=True, epoch=epoch, pool=pool)
                cur_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("lr", cur_lr, global_step=epoch * len(train_loader))
                # if args.resume is not None:
                scheduler.step()
            if not args.train_only:
                run(net, test_loader, None, criterion, writer, args, train=False, epoch=epoch, pool=pool)
                if args.eval_only:
                    exit()

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": vars(args),
        }
        print(os.path.join("logs", args.name))
        torch.save(results, os.path.join("logs", args.name))
        if args.eval_only:
            break


if __name__ == "__main__":
    main()
