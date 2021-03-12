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

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

axislabel_fontsize = 7
ticklabel_fontsize = 7
titlelabel_fontsize = 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


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
        img_saliencies = resize_tensor(img_saliencies.cpu(), 224, 224).squeeze(dim=1).cpu()
        # img_saliencies = generate_inpgrad_captum_img(net, imgs, preds).squeeze(dim=1)

        for img_id, (img, img_expl, true_label, pred_label, imgid) in enumerate(zip(
                imgs, img_saliencies, img_class_ids, preds, img_ids)):
            if img_id > 100:
                break
            # unnormalize images
            img = img / 2. + 0.5  # Rescale to [0, 1].

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")))
            ax.axis('off')
            plt.savefig(f"{save_path}{tagname}_img_{imgid}_true_{true_label}_pred_{pred_label}.png",
                        dpi=300, bbox_inches='tight',
                        frameon=False, pad_inches=0)
            plt.close()

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.imshow(img_expl.detach().cpu().numpy())
            ax.axis('off')
            plt.savefig(f"{save_path}{tagname}_expl_{imgid}.png", dpi=300, bbox_inches='tight',
                        frameon=False, pad_inches=0)
            plt.close()


            fig = create_expl_images(np.array(transforms.ToPILImage()(img.cpu()).convert("RGB")),
                             img_expl.detach().cpu().numpy(),
                             true_label, pred_label)
            plt.savefig(f"{save_path}{tagname}_{imgid}.png")
            plt.close(fig)

        break


def create_writer(args):
    writer = SummaryWriter(f"runs/{args.conf_version}/{args.name}_seed{args.seed}", purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('Hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('Hyperparameters/batchsize', args.batch_size, 0)

    # store args as txt file
    with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")
    return writer


def create_expl_images(img, img_expl, true_class_name, pred_class_name):
    """
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 2))
    ax[0].imshow(img)#(orig_attrs * 255).astype(np.uint8))
    ax[0].axis('off')

    ax[1].imshow(img_expl)#(orig_attrs * 255).astype(np.uint8))
    ax[1].axis('off')

    fig.suptitle(f"True Class: {true_class_name}; Pred Class: {pred_class_name}", fontsize=titlelabel_fontsize)
    # plt.subplots_adjust(left=0., right=1., bottom=0., top=0.9, wspace=0., hspace=0.)

    return fig


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    # print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {:.3f} Recall: {:.3f}, Accuracy: {:.3f}: ,f1_score: {:.3f}'.format(precision*100,recall*100,
                                                                                         accuracy*100,f1_score*100))
    return precision, recall, accuracy, f1_score


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None,
                          cmap=plt.cm.Blues, sFigName='confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
# Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(sFigName)
    return ax


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
