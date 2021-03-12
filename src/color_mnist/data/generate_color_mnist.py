import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import tqdm
import os
from colour import Color

data_path = './data/'

trans = ([transforms.ToTensor()])
trans = transforms.Compose(trans)
fulltrainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=trans)
trainloader = torch.utils.data.DataLoader(fulltrainset, batch_size=2000, shuffle=False, num_workers=2, pin_memory=True)
test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=trans)
testloader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=2, pin_memory=True)
nb_classes = 10


red = Color("red")
COLORS = list(red.range_to(Color("purple"),10))
COLORS = [np.asarray(x.get_rgb())*255 for x in COLORS]
COLORS = [x.astype('int') for x in COLORS]
COLOR_NAMES = ['red', 'tangerine', 'lime', 'harlequin', 'malachite', 'persian green',
                'allports', 'resolution blue', 'pigment indigo', 'purple']


def gen_fgcolor_data(loader, img_size=(3, 28, 28), split='train'):
    tot_iters = len(loader)
    for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
        x, targets = next(iter(loader))
        assert len(
            x.size()) == 4, 'Something is wrong, size of input x should be 4 dimensional (B x C x H x W; perhaps number of channels is degenrate? If so, it should be 1)'
        targets = targets.cpu().numpy()
        bs = targets.shape[0]

        x_rgb = torch.ones(x.size(0), 3, x.size()[2], x.size()[3]).type('torch.FloatTensor')
        x_rgb = x_rgb * x
        x_rgb_fg = 1. * x_rgb

        if split == 'test':
            color_choice = np.random.randint(0, 10, targets.shape[0])
        elif split == 'train':
            color_choice = targets.astype('int')
        c = np.array([COLORS[ind] for ind in color_choice])
        c = c.reshape(-1, 3, 1, 1)
        c = torch.from_numpy(c).type('torch.FloatTensor')
        x_rgb_fg[:, 0] = x_rgb_fg[:, 0] * c[:, 0]
        x_rgb_fg[:, 1] = x_rgb_fg[:, 1] * c[:, 1]
        x_rgb_fg[:, 2] = x_rgb_fg[:, 2] * c[:, 2]

        bg = (torch.zeros_like(x_rgb))
        x_rgb = x_rgb_fg + bg
        x_rgb = torch.clamp(x_rgb, 0., 255.)

        if i == 0:
            color_data_x = np.zeros((bs * tot_iters, *img_size))
            color_data_y = np.zeros((bs * tot_iters,))

        color_data_x[i * bs: (i + 1) * bs] = x_rgb / 255.
        color_data_y[i * bs: (i + 1) * bs] = targets
    return color_data_x, color_data_y


dir_name = data_path + '/cmnist/'
print(dir_name)
if not os.path.exists(data_path + 'cmnist/'):
    os.makedirs(data_path + 'cmnist/', exist_ok=True)
if not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

color_data_x, color_data_y = gen_fgcolor_data(trainloader, img_size=(3, 28, 28), split='train')
np.save(dir_name + 'train_x.npy', color_data_x)
np.save(dir_name + 'train_y.npy', color_data_y)

color_data_x, color_data_y = gen_fgcolor_data(testloader, img_size=(3, 28, 28), split='test')
np.save(dir_name + 'test_x.npy', color_data_x)
np.save(dir_name + 'test_y.npy', color_data_y)
