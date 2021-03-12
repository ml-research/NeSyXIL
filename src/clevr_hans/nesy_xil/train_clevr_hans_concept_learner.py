import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random
import argparse
import glob
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn import metrics
from tqdm import tqdm

import NeSyConceptLearner.src.model as model
import NeSyConceptLearner.src.utils as utils
import data_clevr_hans as data
from rtpt import RTPT

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

# -----------------------------------------
# - Define basic and data related methods -
# -----------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--mode", type=str, required=True, help="train, test, or plot")
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
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
        choices=["clevr-hans-state"],
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument("--data-dir", type=str, help="Directory to data")
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")

    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')

    args = parser.parse_args()

    # hard set !!!!!!!!!!!!!!!!!!!!!!!!!
    args.n_heads = 4
    args.set_transf_hidden = 128

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}"

    if args.mode == 'test':
        assert args.fp_ckpt

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    args.conf_num = args.data_dir.split('Clevr_Hans/')[-1].split('/')[0]

    utils.seed_everything(args.seed)

    return args


def get_confusion_from_ckpt(net, test_loader, criterion, args, datasplit, writer=None):

    true, pred, true_wrong, pred_wrong = run_test_final(net, test_loader, criterion, writer, args, datasplit)
    precision, recall, accuracy, f1_score = utils.performance_matrix(true, pred)

    # Generate Confusion Matrix
    if writer is not None:
        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_normalize_{}.pdf'.format(
                                  datasplit))
                              )
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_{}.pdf'.format(datasplit)))
    else:
        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_normalize_{}.pdf'.format(datasplit)))
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_{}.pdf'.format(datasplit)))
    return accuracy

# -----------------------------------------
# - Define Train/Test/Validation methods -
# -----------------------------------------
def run_test_final(net, loader, criterion, writer, args, datasplit):
    net.eval()

    running_corrects = 0
    running_loss=0
    pred_wrong = []
    true_wrong = []
    preds_all = []
    labels_all = []
    with torch.no_grad():

        for i, sample in enumerate(tqdm(loader)):
            # input is either a set or an image
            imgs, target_set, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
            img_class_ids = img_class_ids.long()

            # forward evaluation through the network
            output_cls, output_attr = net(imgs)
            _, preds = torch.max(output_cls, 1)

            labels_all.extend(img_class_ids.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

            running_corrects = running_corrects + torch.sum(preds == img_class_ids)
            loss = criterion(output_cls, img_class_ids)
            running_loss += loss.item()
            preds = preds.cpu().numpy()
            target = img_class_ids.cpu().numpy()
            preds = np.reshape(preds, (len(preds), 1))
            target = np.reshape(target, (len(preds), 1))

            for i in range(len(preds)):
                if (preds[i] != target[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(target[i])

        bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

        if writer is not None:
            writer.add_scalar(f"Loss/{datasplit}_loss", running_loss / len(loader), 0)
            writer.add_scalar(f"Acc/{datasplit}_bal_acc", bal_acc, 0)

        return labels_all, preds_all, true_wrong, pred_wrong


def run(net, loader, optimizer, criterion, split, writer, args, train=False, plot=False, epoch=0):
    if train:
        net.train()
        # net.set_cls.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        # net.set_cls.eval()
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc="{1} E{0:02d}".format(epoch, "train" if train else "val "),
    )
    running_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        _, preds = torch.max(output_cls, 1)

        loss = criterion(output_cls, img_class_ids)

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        labels_all.extend(img_class_ids.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch):
            utils.write_expls(net, loader, f"Expl/{split}", epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def train(args):
    print("Data loading ...\n")

    if args.dataset == "clevr-hans-state":
        dataset_train = data.CLEVR_HANS_EXPL(
            args.data_dir, "train", lexi=True, conf_vers=args.conf_version
        )
        dataset_val = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", lexi=True, conf_vers=args.conf_version
        )
        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "test", lexi=True, conf_vers=args.conf_version
        )
    else:
        print("Wrong dataset specifier")
        exit()

    print("Data loaded ...\n")

    args.n_imgclasses = dataset_train.n_classes
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = dataset_train.category_ids

    train_loader = data.get_loader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = data.get_loader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)

    # load pretrained state predictor
    log = torch.load("logs/slot-attention-clevr-state-3_final")
    net.img2state_net.load_state_dict(log['weights'], strict=True)
    print("Pretrained slot attention model loaded!")

    net = net.to(args.device)

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(
        [p for name, p in net.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)
    torch.backends.cudnn.benchmark = True

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"Clevr Hans Slot Att Set Transf default",
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    writer = utils.create_writer(args)
    # writer = None

    cur_best_val_loss = np.inf
    for epoch in range(args.epochs):
        _ = run(net, train_loader, optimizer, criterion, split='train', args=args, writer=writer,
                train=True, plot=False, epoch=epoch)
        scheduler.step()
        val_loss = run(net, val_loader, optimizer, criterion, split='val', args=args, writer=writer,
                train=False, plot=True, epoch=epoch)
        _ = run(net, test_loader, optimizer, criterion, split='test', args=args, writer=writer,
                train=False, plot=False, epoch=epoch)

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": args,
        }
        if cur_best_val_loss > val_loss:
            if epoch > 0:
                # remove previous best model
                os.remove(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
            torch.save(results, os.path.join(writer.log_dir, "model_epoch{}_bestvalloss_{:.4f}.pth".format(epoch,
                                                                                                           val_loss)))
            cur_best_val_loss = val_loss

        # Update the RTPT (subtitle is optional)
        rtpt.step()


    # load best model for final evaluation
    net = model.IMG2TabCls(args, n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
                           category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)
    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best',
                            writer=writer)
    get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
                            writer=writer)

    # plot expls
    run(net, train_loader, optimizer, criterion, split='train_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    run(net, val_loader, optimizer, criterion, split='val_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    run(net, test_loader, optimizer, criterion, split='test_best', args=args, writer=writer, train=False, plot=True, epoch=0)

    writer.close()


def test(args):

    print(f"\n\n{args.name} seed {args.seed}\n")

    if args.dataset == "clevr-hans-state":
        dataset_val = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", lexi=True, conf_vers=args.conf_version
        )
        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "test", lexi=True, conf_vers=args.conf_version
        )
    else:
        print("Wrong dataset specifier")
        exit()

    args.n_imgclasses = dataset_val.n_classes
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = dataset_val.category_ids

    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = data.get_loader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    # load best model for final evaluation
    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)

    checkpoint = torch.load(args.fp_ckpt)
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    acc = get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best', writer=None)
    print(f"\nVal. accuracy: {(100*acc):.2f}")
    acc = get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best', writer=None)
    print(f"\nTest accuracy: {(100*acc):.2f}")


def plot(args):

    print(f"\n\n{args.name} seed {args.seed}\n")

    if args.dataset == "clevr-hans-state":
        dataset_val = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", lexi=True, conf_vers=args.conf_version
        )
        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "test", lexi=True, conf_vers=args.conf_version
        )
    else:
        print("Wrong dataset specifier")
        exit()

    args.n_imgclasses = dataset_val.n_classes
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = dataset_val.category_ids

    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # load best model for final evaluation
    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)

    checkpoint = torch.load(args.fp_ckpt)
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    save_dir = args.fp_ckpt.split('model_epoch')[0]+'figures/'
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # directory already exists
        pass

    # change plotting function in utils in order to visualize explanations
    assert args.conf_version == 'CLEVR-Hans3'
    utils.save_expls_attn(net, test_loader, "test", save_path=save_dir)


def main():
    args = get_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'plot':
        plot(args)


if __name__ == "__main__":
    main()
