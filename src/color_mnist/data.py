import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils

torch.manual_seed(0)

NUM_WORKERS = 2


def get_dataset(root, dataset, batch_size):
    if dataset == 'mnist':
        trans = ([transforms.ToTensor()])
        trans = transforms.Compose(trans)
        fulltrainset = torchvision.datasets.MNIST(root=root, train=True, transform=trans, download=True)

        train_set = fulltrainset
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                  num_workers=NUM_WORKERS, pin_memory=True)
        #validloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
        #                                          num_workers=NUM_WORKERS, pin_memory=True)
        validloader = None
        test_set = torchvision.datasets.MNIST(root=root, train=False, transform=trans)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        nb_classes = 10
        dim_inp = 28 * 28
    elif 'cmnist' in dataset:
        data_dir_cmnist = root + dataset + '/'
        data_x = np.load(data_dir_cmnist + 'train_x.npy')
        data_y = np.load(data_dir_cmnist + 'train_y.npy')

        data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
        data_y = torch.from_numpy(data_y).type('torch.LongTensor')

        my_dataset = utils.TensorDataset(data_x, data_y)

        #train_set, valset = _split_train_val(my_dataset, val_fraction=0.1)
        train_set = my_dataset
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        #validloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
        #                                          num_workers=NUM_WORKERS, pin_memory=True)
        validloader = None
        data_x = np.load(data_dir_cmnist + 'test_x.npy')
        data_y = np.load(data_dir_cmnist + 'test_y.npy')
        data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
        data_y = torch.from_numpy(data_y).type('torch.LongTensor')
        my_dataset = utils.TensorDataset(data_x, data_y)
        testloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        nb_classes = 10
        dim_inp = 28 * 28 * 3
    else:
        raise ValueError('unknown dataset')
    return trainloader, validloader, testloader, nb_classes, dim_inp


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     #trainloader, validloader, testloader, nb_classes, dim_inp = get_dataset('./.data', 'fgbg_cmnist_cpr0.5-0.5', 1)
#     trainloader, validloader, testloader, nb_classes, dim_inp = get_dataset('./.data', 'fgbg_cmnist_cpr0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1-0.1', 1)
#     #trainloader, validloader, testloader, nb_classes, dim_inp = get_dataset('./.data', 'fgbg_cmnist_cpr1', 1)
#
#     for i in range(10):
#         cnt = 0
#         for sample, target in trainloader:
#             if target.item() == i:
#                 plt.figure()
#                 sample = np.transpose(sample.cpu().detach().numpy().squeeze(), (1,2,0))
#                 plt.imshow(sample)
#                 cnt += 1
#             if cnt == 4:
#                 cnt = 0
#                 break
#         plt.show()
