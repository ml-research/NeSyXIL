import os
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.utils as torchvision_utils
import torchvision.datasets as datasets
import torchvision.models as models
import torch.multiprocessing as mp

import scipy.optimize
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from captum.attr import InputXGradient, IntegratedGradients, DeepLift, NoiseTunnel
from captum.attr._core.layer.grad_cam import LayerGradCam

import data_xil as data
import utils as utils
import model as model
from rrr_loss import rrr_loss_function

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
    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument("--resume", help="Path to log file to resume from")
    parser.add_argument("--mode", type=str, required=True, help="train, test, or plot")
    parser.add_argument("--data-dir", type=str, help="Directory to data")
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--l2_grads", type=float, default=1, help="Right for right reason weight"
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
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )

    args = parser.parse_args()

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}"

    if args.mode == 'test':
        assert args.fp_ckpt

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

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
            output_cls = net(imgs)
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

        print("Test Accuracy: {:.4f}".format(bal_acc))
        if writer is not None:
            writer.add_scalar(f"Loss/{datasplit}_loss", running_loss / len(loader), 0)
            writer.add_scalar(f"Acc/{datasplit}_bal_acc", bal_acc, 0)

        return labels_all, preds_all, true_wrong, pred_wrong


def run(net, loader, optimizer, criterion, split, writer, args, train=False, plot=False, epoch=0):
    if train:
        net.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
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
        imgs, _, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls = net(imgs)
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
            utils.write_expls(net, loader, f"Expl/{split}", args, epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def run_lexi(net, loader, optimizer, criterion, criterion_lexi, split, writer, args, train=True, plot=False, epoch=0):

    def save_target_output(self, input, output):
        net.target_output = output

    def forward_pass_on_convolutions(x, target_layer):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        net.features[-1].register_forward_hook(save_target_output)

    forward_pass_on_convolutions(net, target_layer=None)

    net.train()
    torch.set_grad_enabled(True)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc="{1} E{0:02d}".format(epoch, "train" if train else "val "),
    )
    running_loss = 0
    running_ra_loss = 0
    running_lexi_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        # input is either a set or an image
        imgs, _, img_class_ids, img_ids, img_expls, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls = net(imgs)
        _, preds = torch.max(output_cls, 1)

        loss_pred = criterion(output_cls, img_class_ids)
        if 'val' in split:
            loss_lexi = criterion_lexi(net, imgs, img_class_ids, img_expls, epoch, batch_id=i, writer=writer,
                                       writer_prefix=split)
        else:
            loss_lexi = criterion_lexi(net, imgs, img_class_ids, img_expls, epoch, batch_id=i, writer=None,
                                       writer_prefix=split)

        loss = loss_pred + loss_lexi

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_ra_loss += loss_pred.item()
        running_lexi_loss += loss_lexi.item()/ args.l2_grads

        labels_all.extend(img_class_ids.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch):
            utils.write_expls(net, loader, f"Expl/{split}", args, epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Loss/{split}_ra_loss", running_ra_loss / len(loader), epoch)
    writer.add_scalar(f"Loss/{split}_lexi_loss", running_lexi_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.4f}.. ".format(split, running_loss / len(loader)),
          "{} RA Loss: {:.4f}.. ".format(split, running_ra_loss / len(loader)),
          "{} Lexi Loss: {:.4f}.. ".format(split, running_lexi_loss / len(loader)),
          "{} Accuracy: {:.4f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


class LexiLoss:
    def __init__(self, alpha, class_weights, args):
        self.alpha = alpha
        self.lambda_sparse = 10.
        self.criterion = torch.nn.MSELoss()
        self.args = args

    def resize_tensor(self, input_tensors, h, w):
        input_tensors = torch.squeeze(input_tensors, 1)

        for i, img in enumerate(input_tensors):
            img_PIL = transforms.ToPILImage()(img)
            img_PIL = transforms.Resize([h, w], interpolation=1)(img_PIL)
            img_PIL = transforms.ToTensor()(img_PIL)
            if i == 0:
                final_output = img_PIL
            else:
                final_output = torch.cat((final_output, img_PIL), 0)
        final_output = torch.unsqueeze(final_output, 1)
        return final_output

    def norm_saliencies(self, saliencies):
        saliencies_norm = saliencies.clone()

        for i in range(saliencies.shape[0]):
            if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
                saliencies_norm[i] = saliencies[i]
            else:
                saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                     (torch.max(saliencies[i]) - torch.min(saliencies[i]))

        return saliencies_norm

    def generate_gradcam_captum(self, model, x, class_ids):
        class_ids = class_ids.to("cuda")
        explainer = LayerGradCam(model, model.features[-1])
        saliencies = explainer.attribute(x, target=class_ids,
                                         relu_attributions=True)
        return self.norm_saliencies(saliencies)

    def __call__(self, model, imgs, class_ids, masks, epoch, batch_id, writer=None, writer_prefix=""):

        masks = masks.squeeze(dim=1)
        saliencies = self.generate_gradcam_captum(model, imgs, class_ids).squeeze(dim=1)

        assert len(saliencies.shape) == 3
        assert masks.shape == saliencies.shape

        # captum gradcam is size of conv layer --> must resize gt mask
        loss = self.criterion(saliencies, masks)

        if writer is not None and batch_id == 0:
            # upscale saliencies from captum gradcam
            saliencies = self.resize_tensor(saliencies.cpu(), 224, 224).cpu()
            masks = self.resize_tensor(masks.cpu(), 224, 224).cpu()

            for i in range(imgs.shape[0]):
                if epoch <= 1:

                    img = imgs[i]
                    # unnormalize images
                    img = img / 2. + 0.5  # Rescale to [0, 1].

                    writer.add_image('{}_{}images'.format(writer_prefix, i), imgs[i], epoch)
                    writer.add_image('{}_{}masks'.format(writer_prefix, i), masks[i], epoch)
                writer.add_image('{}_{}saliencies'.format(writer_prefix, i), saliencies[i], epoch)

        loss = self.alpha * loss
        return loss

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
    # Clevr Hans dataset is balanced
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
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

    net = model.ResNet34Small(num_classes=args.n_imgclasses)
    net = net.to(args.device)

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion_lexi = LexiLoss(alpha=args.l2_grads, class_weights=args.class_weights.float().to("cuda"), args=args)

    torch.backends.cudnn.benchmark = True

    writer = utils.create_writer(args)

    cur_best_val_loss = np.inf
    _ = run_lexi(net, val_loader, optimizer, criterion, criterion_lexi, split='val_prior', args=args, writer=writer,
                 train=False, plot=False, epoch=0)
    for epoch in range(args.epochs):
        _ = run_lexi(net, train_loader, optimizer, criterion, criterion_lexi, split='train', args=args,
                             writer=writer, train=True, plot=False, epoch=epoch)
        val_loss = run_lexi(net, val_loader, optimizer, criterion, criterion_lexi, split='val', args=args, writer=writer,
                    train=False, plot=False, epoch=epoch)
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

    # load best model for final evaluation
    net = model.ResNet34Small(num_classes=args.n_imgclasses)
    net = net.to(args.device)
    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_epoch*_bestvalloss*.pth"))[0])
    # load best model for final evaluation
    net.load_state_dict(checkpoint['weights'])
    print("\nModel loaded from checkpoint for final evaluation\n")

    get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test',
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
    # Clevr Hans dataset is balanced
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
    criterion_lexi = LexiLoss(alpha=args.l2_grads, class_weights=args.class_weights.float().to("cuda"), args=args)

    # load best model for final evaluation
    net = model.ResNet34Small(num_classes=args.n_imgclasses)
    net = net.to(args.device)
    checkpoint = torch.load(args.fp_ckpt)
    # load best model for final evaluation
    net.load_state_dict(checkpoint['weights'])
    print("\nModel loaded from checkpoint for final evaluation\n")

    acc = get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best', writer=None)
    print(f"\nVal. accuracy: {(100 * acc):.2f}")
    acc = get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test', writer=None)
    print(f"\nTest accuracy: {(100 * acc):.2f}")


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
    val_loader = data.get_loader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    # load best model for final evaluation
    net = model.ResNet34Small(num_classes=args.n_imgclasses)
    net = net.to(args.device)
    checkpoint = torch.load(args.fp_ckpt)
    # load best model for final evaluation
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    save_dir = args.fp_ckpt.split('model_epoch')[0]+'figures/'
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # directory already exists
        pass

    utils.save_expl_images(net, test_loader, "test", save_path=save_dir)


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
