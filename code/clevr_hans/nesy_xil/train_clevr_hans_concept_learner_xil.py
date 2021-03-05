import os
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
import model
import utils as utils
from rtpt import RTPT

torch.autograd.set_detect_anomaly(True)

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
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")

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

    if args.mode == 'test' or args.mode == 'plot':
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


def write_expls_attn(net, data_loader, tagname, args, epoch, writer):

    def norm_saliencies(saliencies):
        saliencies_norm = saliencies.clone()

        for i in range(saliencies.shape[0]):
            if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
                saliencies_norm[i] = saliencies[i]
            else:
                saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                     (torch.max(saliencies[i]) - torch.min(saliencies[i]))

        return saliencies_norm

    def generate_intgrad_captum_table(net, input, labels):
        labels = labels.to("cuda")
        explainer = IntegratedGradients(net)
        saliencies = explainer.attribute(input, target=labels)
        # remove negative attributions
        saliencies[saliencies < 0] = 0.
        # normalize each saliency map by its max
        for k, sal in enumerate(saliencies):
            saliencies[k] = sal/torch.max(sal)
        return norm_saliencies(saliencies)

    attr_labels = ['Sphere', 'Cube', 'Cylinder',
                   'Large', 'Small',
                   'Rubber', 'Metal',
                   'Cyan', 'Blue', 'Yellow', 'Purple', 'Red', 'Green', 'Gray', 'Brown']

    net.eval()

    for i, sample in enumerate(data_loader):
        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, _ = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        _, preds = torch.max(output_cls, 1)

        # convert sorting gt target set and gt table explanations to match the order of the predicted table
        target_set, match_ids = utils.hungarian_matching(output_attr.to('cuda'), target_set)
        # table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        # get explanations of set classifier
        table_saliencies = generate_intgrad_captum_table(net.set_cls, output_attr, preds)

        # get the ids of the two objects that receive the maximal importance, i.e. most important for the classification
        max_expl_obj_ids = table_saliencies.max(dim=2)[0].topk(2)[1]

        # get attention masks
        attns = net.img2state_net.slot_attention.attn
        # reshape attention masks to 2D
        attns = attns.reshape((attns.shape[0], attns.shape[1], int(np.sqrt(attns.shape[2])),
                               int(np.sqrt(attns.shape[2]))))

        # concatenate the visual explanation of the top two objects that are most important for the classification
        img_saliencies = torch.zeros(attns.shape[0], attns.shape[2], attns.shape[3])
        # for obj_id in range(max_expl_obj_ids.shape[1]):
        #     # convert the ids of the max expl ids to the corresponding slot ids, that were computed by
        #     # utils.hungarian_matching
        #     convert_ids = match_ids[range(attns.shape[0]), max_expl_obj_ids[:, obj_id].cpu().numpy()]
        #     img_saliencies += attns[range(attns.shape[0]), convert_ids, :, :].detach().cpu()
        for obj_id in range(max_expl_obj_ids.shape[1]):
            img_saliencies += attns[range(attns.shape[0]), obj_id, :, :].detach().cpu()

        # upscale img_saliencies to orig img shape
        # imgs = utils.resize_tensor(imgs.cpu(), 3, 300, 300).squeeze(dim=1).cpu()
        img_saliencies = utils.resize_tensor(img_saliencies.cpu(), imgs.shape[2], imgs.shape[2]).squeeze(dim=1).cpu()

        for img_id, (img, gt_table, pred_table, table_expl, img_expl, true_label, pred_label, imgid) in enumerate(zip(
                imgs, target_set, output_attr, table_saliencies,
                img_saliencies, img_class_ids, preds,
                img_ids
        )):
            # unnormalize images
            img = img / 2. + 0.5  # Rescale to [0, 1].

            # fig = utils.create_expl_images(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")),
            #                 gt_table.detach().cpu().numpy(),
            #                  pred_table.detach().cpu().numpy(),
            #                  table_expl.detach().cpu().numpy(),
            #                  img_expl.detach().cpu().numpy(),
            #                  true_label, pred_label,
            #                  attr_labels)
            fig = utils.create_expl_images(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")),
                                           pred_table.detach().cpu().numpy(),
                                           table_expl.detach().cpu().numpy(),
                                           img_expl.detach().cpu().numpy(),
                                           true_label, pred_label, attr_labels)
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
            imgs, target_set, img_class_ids, img_ids, _, table_expl = map(lambda x: x.cuda(), sample)
            img_class_ids = img_class_ids.long()

            # forward evaluation through the network
            # output_cls, output_attr = net(imgs)
            output_cls = net.set_cls(target_set)
            # class prediction
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
        net.img2state_net.eval()
        net.set_cls.train()
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
        imgs, target_set, img_class_ids, img_ids, _, table_expl = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        # output_cls, output_attr = net(imgs)
        output_cls = net.set_cls(target_set)
        # class prediction
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
        # if plot and i == 0:
        #     write_expls(net, loader, f"Expl/{split}", args, epoch, writer)
            write_expls_attn(net, loader, f"Expl/{split}", args, epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def run_lexi(net, loader, optimizer, criterion, criterion_lexi, split, writer, args, train=True, plot=False, epoch=0):

    if train:
        net.img2state_net.eval()
        net.set_cls.train()
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

    # Plot initial predictions in Tensorboard
    if plot and epoch == 0:
        write_expls_attn(net, loader, f"Expl/prior_{split}", args, 0, writer)

    running_loss = 0
    running_ra_loss = 0
    running_lexi_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):

        # input is either a set or an image
        imgs, target_set, img_class_ids, img_ids, _, table_expls = map(lambda x: x.cuda(), sample)
        img_class_ids = img_class_ids.long()

        # # sanity check slot attention
        # output = net.img2state_net(imgs)
        # output = net.img2state_net._transform_attrs(output)
        # # print predictions for one image, match predictions with targets
        # matched_output, _ = utils.hungarian_matching(target_set, output.to('cuda'), verbose=0)
        # for k in range(2):
        #     print(f"\nGT: \n{target_set.detach().cpu().numpy()[k]}")
        #     # print(f"\nPred: \n{np.round(matched_output.detach().cpu().numpy()[k], 0)}\n")
        #     print(f"\nPred: \n{matched_output.detach().cpu().numpy()[k]}\n")
        # exit()

        # forward evaluation through the network
        output_cls, output_attr = net(imgs)
        _, preds = torch.max(output_cls, 1)

        # convert sorting gt target set and gt table explanations to match the order of the predicted table
        target_set, match_ids = utils.hungarian_matching(output_attr.to('cuda'), target_set)
        table_expls = table_expls[:, match_ids][range(table_expls.shape[0]), range(table_expls.shape[0])]

        loss_pred = criterion(output_cls, img_class_ids)

        loss_lexi = criterion_lexi(net.set_cls, output_attr, img_class_ids, table_expls, epoch, batch_id=i,
                                   writer=None, writer_prefix=split)

        # if epoch < 10:
        #     l2_grads = 0
        # else:
        l2_grads = args.l2_grads
        loss = loss_pred + l2_grads * loss_lexi

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
            write_expls_attn(net, loader, f"Expl/{split}", args, epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    if split == "train":
        writer.add_scalar(f"L2_grads/{split}_l2_grads", l2_grads, epoch)
    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Loss/{split}_ra_loss", running_ra_loss / len(loader), epoch)
    writer.add_scalar(f"Loss/{split}_lexi_loss", running_lexi_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("{} Loss: {:.4f}.. ".format(split, running_loss / len(loader)),
          "{} RA Loss: {:.4f}.. ".format(split, running_ra_loss / len(loader)),
          "{} Lexi Loss: {:.4f}.. ".format(split, running_lexi_loss / len(loader)),
          "{} Accuracy: {:.4f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


class LexiLoss:
    def __init__(self, class_weights, args):
        self.criterion = torch.nn.MSELoss()
        self.args = args

    def norm_saliencies(self, saliencies):
        saliencies_norm = saliencies.clone()

        for i in range(saliencies.shape[0]):
            if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
                saliencies_norm[i] = saliencies[i]
            else:
                saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                     (torch.max(saliencies[i]) - torch.min(saliencies[i]))

        return saliencies_norm

    def generate_intgrad_captum_table(self, net, input, labels):
        labels = labels.to("cuda")
        explainer = IntegratedGradients(net)
        saliencies = explainer.attribute(input, target=labels)
        # remove negative attributions
        saliencies[saliencies < 0] = 0.
        return self.norm_saliencies(saliencies)

    def __call__(self, net, target_set, class_ids, masks, epoch, batch_id, writer=None, writer_prefix=""):

        masks = masks.squeeze(dim=1).double()
        saliencies = self.generate_intgrad_captum_table(net, target_set, class_ids).squeeze(dim=1)

        assert len(saliencies.shape) == 3
        assert masks.shape == saliencies.shape

        loss = self.criterion(saliencies, masks)

        # if writer is not None and batch_id == 0:
        #     print("Creating figures...")
        #
        #     for i in range(target_set.shape[0]):
        #         if epoch <= 1:
        #             writer.add_image('{}_{}gt_tables'.format(writer_prefix, i), target_set[i].unsqueeze(0), epoch)
        #             writer.add_image('{}_{}masks'.format(writer_prefix, i), masks[i].unsqueeze(0), epoch)
        #         writer.add_image('{}_{}saliencies'.format(writer_prefix, i), saliencies[i].unsqueeze(0), epoch)

        return loss


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

    net = model.IMG2TabCls(args, n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
                           set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
                           device=args.device)

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
    criterion_lexi = LexiLoss(class_weights=args.class_weights.float().to("cuda"), args=args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    torch.backends.cudnn.benchmark = True

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"Clevr Hans Slot Att Set Transf xil",
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # tensorboard writer
    writer = utils.create_writer(args)
    # writer = None

    cur_best_val_loss = np.inf
    for epoch in range(args.epochs):
        _ = run_lexi(net, train_loader, optimizer, criterion, criterion_lexi, split='train', args=args,
                             writer=writer, train=True, plot=False, epoch=epoch)
        scheduler.step()
        val_loss = run_lexi(net, val_loader, optimizer, criterion, criterion_lexi, split='val', args=args,
                            writer=writer, train=False, plot=True, epoch=epoch)
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
                           set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
                           device=args.device)
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

    net = model.IMG2TabCls(args, n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
                           set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
                           device=args.device)
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

    # no positional info per object
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
    net = model.IMG2TabCls(args, n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
                           set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
                           device=args.device)
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
    assert args.conf_version == 'conf_3'
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
