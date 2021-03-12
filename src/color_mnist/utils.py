import numpy as np
import random
import io
import os
import torch
import matplotlib.pyplot as plt
# from skimage import color
from sklearn import metrics
from matplotlib import rc
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from captum.attr._core.layer.grad_cam import LayerGradCam
from matplotlib.colors import ListedColormap
from matplotlib import cm

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

axislabel_fontsize = 7
ticklabel_fontsize = 7
titlelabel_fontsize = 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def get_white_viridis_cmap(range_=128):
    viridis = cm.get_cmap('viridis', 256)
    viridis_colors = viridis.colors.copy()
    viridis_white = viridis_colors[range_ + 1].copy()
    viridis_colors[0:range_, 0] = np.linspace(1, viridis_white[0], range_)
    viridis_colors[0:range_, 1] = np.linspace(1, viridis_white[1], range_)
    viridis_colors[0:range_, 2] = np.linspace(1, viridis_white[2], range_)
    viridis_colors[0:range_, 3] = np.linspace(1, viridis_white[3], range_)

    newcmp = ListedColormap(viridis_colors)
    return newcmp


def save_expl_images(net, data_loader, tagname, save_path):

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

    for i, sample in enumerate(data_loader):

        # input is either a set or an image
        imgs, targets = map(lambda x: x.cuda(), sample)

        # forward evaluation through the network
        output = net(imgs)
        preds = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

        # get explanations of image encoder
        img_saliencies = generate_gradcam_captum_img(net, imgs, preds).squeeze(dim=1)
        img_saliencies = resize_tensor(img_saliencies.cpu(), 28, 28).squeeze(dim=1).cpu()
        # img_saliencies = generate_inpgrad_captum_img(net, imgs, preds).squeeze(dim=1)

        for img_id, (img, img_expl, true_label, pred_label) in enumerate(zip(
                imgs, img_saliencies, targets, preds.detach().cpu().numpy())):
            if img_id == 13:

                img = np.array(transforms.ToPILImage()(img.cpu()).convert("RGB"))
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.imshow(img)
                ax.axis('off')
                plt.savefig(f"{save_path}{tagname}_img_{img_id}_true_{true_label}_pred_{pred_label}.png",
                            dpi=300, bbox_inches='tight',
                            frameon=False, pad_inches=0)
                plt.close()

                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.imshow(img_expl.detach().cpu().numpy())
                ax.axis('off')
                plt.savefig(f"{save_path}{tagname}_expl_{img_id}.png", dpi=300, bbox_inches='tight',
                            frameon=False, pad_inches=0)
                plt.close()

                np.save(f"{save_path}{tagname}_expl_{img_id}_true_{true_label}_pred_{pred_label}.npy", img)
                np.save(f"{save_path}{tagname}_expl_{img_id}.npy", img_expl)

                exit()
        break


def create_overview_plots(trainloader, testloader, save_dir):
    train_x, train_y = next(iter(trainloader))
    fig, ax = plt.subplots(nrows=3, ncols=10, figsize=(10, 3))
    for class_id in range(10):
        tmp = train_x[train_y == class_id]
        assert tmp.shape[0] >= 3
        for i in range(3):
            img = np.moveaxis(np.array(tmp[i]), [0, 1, 2], [2, 0, 1])
            ax[i, class_id].imshow(img)
            ax[i, class_id].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"{save_dir}train_imgs.png", dpi=300, bbox_inches='tight', frameon=False, pad_inches=0)
    plt.close()

    test_x, test_y = next(iter(testloader))
    fig, ax = plt.subplots(nrows=2, ncols=10, figsize=(10, 2))
    for class_id in range(10):
        tmp = test_x[test_y == class_id]
        assert tmp.shape[0] >= 2
        for i in range(2):
            img = np.moveaxis(np.array(tmp[i]), [0, 1, 2], [2, 0, 1])
            ax[i, class_id].imshow(img)
            ax[i, class_id].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"{save_dir}test_imgs.png", dpi=300, bbox_inches='tight', frameon=False, pad_inches=0)
    plt.close()



def resize_tensor(input_tensors, h, w):
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
