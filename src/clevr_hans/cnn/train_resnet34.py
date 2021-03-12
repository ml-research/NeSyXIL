import os
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
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

# -----------------------------------------
# - Define basic and data related methods -
# -----------------------------------------
class ResNet34Small(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Small, self).__init__()
        original_model = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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

    seed_everything(args.seed)

    return args


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_expls(net, data_loader, tagname, args, epoch, writer):
    def norm_saliencies(saliencies):
        saliencies_norm = saliencies.clone()

        for i in range(saliencies.shape[0]):
            saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                 (torch.max(saliencies[i]) - torch.min(saliencies[i]))

        return saliencies_norm

    def generate_gradcam_captum_img(net, data, labels):
        labels = labels.to("cuda")
        explainer = LayerGradCam(net, net.features[-1])
        saliencies = explainer.attribute(inputs=data, target=labels, relu_attributions=True)
        saliencies = norm_saliencies(saliencies)
        return saliencies

    def generate_inpgrad_captum_img(net, data, labels):
        labels = labels.to("cuda")
        explainer = InputXGradient(net)
        saliencies = explainer.attribute(inputs=data, target=labels)
        # sum over rgb channels
        saliencies = torch.sum(saliencies, dim=1)

        saliencies = norm_saliencies(saliencies)
        return saliencies

    attr_labels = ['Sphere', 'Cube', 'Cylinder',
                   'Large', 'Small',
                   'Rubber', 'Metal',
                   'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown']

    net.eval()

    for i, sample in enumerate(data_loader):
        # input is either a set or an image
        imgs, _, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls = net(imgs)
        _, preds = torch.max(output_cls, 1)

        # get explanations of image encoder
        img_saliencies = generate_gradcam_captum_img(net, imgs, preds).squeeze(dim=1)
        img_saliencies = utils.resize_tensor(img_saliencies.cpu(), 224, 224).squeeze(dim=1).cpu()
        # img_saliencies = generate_inpgrad_captum_img(net, imgs, preds).squeeze(dim=1)

        for img_id, (img, img_expl, true_label, pred_label, imgid) in enumerate(zip(
                imgs, img_saliencies, img_class_ids, preds, img_ids)):
            # unnormalize images
            img = img/2. + 0.5  # Rescale to [0, 1].
            fig = utils.create_expl_images(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")),
                             img_expl.detach().cpu().numpy(),
                             true_label, pred_label)
            writer.add_figure(f"{tagname}_{img_id}", fig, epoch)
            if img_id > 10:
                break

        break


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
            imgs, _, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
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

        # store explanations in writer
        if plot and not(i % iters_per_epoch):
        # if plot and i == 0:
            write_expls(net, loader, f"Expl/{split}", args, epoch, writer)
            # write_expls_attn(net, loader, f"Expl/{split}", args, epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def train(args):

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

    net = ResNet34Small(num_classes=args.n_imgclasses)
    net = net.to(args.device)

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    torch.backends.cudnn.benchmark = True

    writer = utils.create_writer(args)

    cur_best_val_loss = np.inf
    for epoch in range(args.epochs):
        _ = run(net, train_loader, optimizer, criterion, split='train', args=args, writer=writer,
                train=True, plot=False, epoch=epoch)
        val_loss = run(net, val_loader, optimizer, criterion, split='val', args=args, writer=writer,
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
    net = ResNet34Small(num_classes=args.n_imgclasses)
    net = net.to(args.device)
    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_epoch*.pth"))[0])
    # load best model for final evaluation
    net.load_state_dict(checkpoint['weights'])
    print("\nModel loaded from checkpoint for final evaluation\n")

    get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test',
                            writer=writer)
    get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
                            writer=writer)

    # plot expls
    run(net, val_loader, optimizer, criterion, split='val_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    run(net, test_loader, optimizer, criterion, split='test', args=args, writer=writer, train=False, plot=True, epoch=0)

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

    torch.backends.cudnn.benchmark = True

    # load best model for final evaluation
    net = ResNet34Small(num_classes=args.n_imgclasses)
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
    net = ResNet34Small(num_classes=args.n_imgclasses)
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
