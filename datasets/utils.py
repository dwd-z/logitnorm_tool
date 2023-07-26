import torchvision.transforms as trn
import torchvision.datasets as dset
from datasets.cifar import CIFAR10, CIFAR100
from datasets.mnist import MNIST
import datasets.svhn_loader as svhn
import datasets.gaussian_generator as gsn
import torch

def build_dataset(dataset, mode="train", size=32, channels=3, 
                    mean=(0.492, 0.482, 0.446), std=(0.247, 0.244, 0.262)):
    '''
    This function builds a specified dataset and defines preprocessing steps. The dataset is 
    supposed to be fed into a data loader.

    Parameters
    ----------
    dataset : str
        The name of the dataset. Can be chosen from 'cifar10', 'cifar100', 'Textures', 'SVHN',
        'Places365', 'LSUN-C', 'LSUN-R', 'GTSRB', 'MNIST', 'Gaussian'.
    mode : str, optional
        The purpose of use of the dataset. Can be chosen from 'train' and 'test'. The default 
        is "train".
    size : int, optional
        The size of input of the model. The default is 32.
    channels : int, optional
        The channels of input of the model. The default is 3. 
    mean : tuple of float, optional
        Mean value for normalization of input images. The default is (0.492, 0.482, 0.446).
    std : tuple of float, optional
        Standard deviation for normalization of input images. The default is (0.247, 0.244, 0.262).

    Returns
    -------
    data : torch.utils.data.Dataset
        Input image data loaded from a folder or downloaded from the website.

    '''
    mean = mean 
    std = std 
    if type(size) is not tuple:
        size = (size, size)
    if channels == 3:
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4),
                                       trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.Resize(size), trn.ToTensor(), trn.Normalize(mean, std)])
    else:
        mean = mean[1]
        std = std[1]
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4),
                                       trn.Grayscale(1), trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.Resize(size), trn.Grayscale(1), trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset == 'cifar10':
        if mode == "train":
            data = CIFAR10(root='./data/',
                                    download=True,
                                    dataset_type="train",
                                    transform=train_transform
                                    )
        else:
            data = CIFAR10(root='./data/',
                                   download=True,
                                   dataset_type="test",
                                   transform=test_transform
                                   )
    elif dataset == 'cifar100':
        if mode == "train":
            data = CIFAR100(root='./data/',
                                     download=True,
                                     dataset_type="train",
                                     transform=train_transform
                                     )
        else:
            data = CIFAR100(root='./data/',
                                    download=True,
                                    dataset_type="test",
                                    transform=test_transform
                                    )
    elif dataset == "Textures":
        if channels == 3:
            data = dset.ImageFolder(root="./data/ood_test/dtd/images/",
                                    transform=trn.Compose([trn.Resize(size), trn.CenterCrop(size),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
        else:
            data = dset.ImageFolder(root="./data/ood_test/dtd/images/",
                                    transform=trn.Compose([trn.Resize(size), trn.CenterCrop(size),
                                                           trn.Grayscale(1), trn.ToTensor(), trn.Normalize(mean, std)]))
    elif dataset == "SVHN":
        if mode == "train":
                data = svhn.SVHN(root='./data/ood_test/svhn/', split="train",
                                transform=train_transform,
                                download=False)
        else:
            data = svhn.SVHN(root='./data/ood_test/svhn/', split="test",
                             transform=test_transform,
                             download=True)

    elif dataset == "Places365": # TODO
        if channels == 3:
            transforms = trn.Compose([trn.Resize(size), trn.CenterCrop(size),
                                                       trn.ToTensor(), trn.Normalize(mean, std)])
        else:
            transforms = trn.Compose([trn.Resize(size), trn.CenterCrop(size),
                                                       trn.Grayscale(1), trn.ToTensor(), trn.Normalize(mean, std)])
        data = dset.ImageFolder(root="./data/ood_test/places365/test_subset/",
                                transform=transforms)

    elif dataset == "LSUN-C":
        if channels == 3:
            transforms = trn.Compose([trn.CenterCrop(size), trn.ToTensor(), trn.Normalize(mean, std)])
        else:
            transforms = trn.Compose([trn.CenterCrop(size), trn.Grayscale(1), trn.ToTensor(), trn.Normalize(mean, std)])
        data = dset.ImageFolder(root="./data/ood_test/LSUN_C/",
                                    transform=transforms)

    elif dataset == "LSUN-R":
        data = dset.ImageFolder(root="./data/ood_test/LSUN_R/",
                                    transform=test_transform)

    elif dataset == "iSUN": #TODO
        data = dset.ImageFolder(root="./data/ood_test/iSUN/",
                                    transform=test_transform)
                                    
    elif dataset == "GTSRB":
        if mode == "train":
            data = dset.ImageFolder(root="./data/GTSRB-Training_fixed/GTSRB/Training/",
                                    transform=train_transform)
        else:
            data = dset.ImageFolder(root="./data/ood_test/GTSRB_Online-Test-Images-Sorted/GTSRB/Online-Test-sort/",
                                    transform=test_transform)

    elif dataset == "MNIST":
        if channels == 3:
            train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(size, padding=4),
                                        trn.Grayscale(3), trn.ToTensor(), trn.Normalize(mean, std)])
            test_transform = trn.Compose([trn.Resize(size), trn.Grayscale(3), trn.ToTensor(), trn.Normalize(mean, std)])

        if mode == "train":     
            data = MNIST(root = "./data/ood_test/MNIST/", 
                                          train = True, 
                                          transform = train_transform,
                                          download=True)
        else:
            data = MNIST(root = "./data/ood_test/MNIST/", 
                                          train = False, 
                                          transform = test_transform,
                                          download=True)        

    elif dataset == "Gaussian":
        data = gsn.GAUSSIAN(root = "./data/ood_test/GAUSSIAN/", 
                                          n_samples=5000, size=size, channels=channels, std=1, transform=trn.Compose([trn.Resize(size), trn.ToTensor(), trn.Normalize(mean, std)]))     

    return data


def get_ood_dataloaders(ood_data_list, input_size, input_channels, mean, std, test_bs=200, prefetch_threads=4, seed=1):
    '''
    This function returns an array of OOD data loaders given the name of the datasets.

    Parameters
    ----------
    ood_data_list : str
        An array of name of the OOD datasets. Can be chosen from 'cifar10', 'cifar100', 'Textures', 'SVHN',
        'Places365', 'LSUN-C', 'LSUN-R', 'GTSRB', 'MNIST', 'Gaussian'.
    input_size : int
        The size of input of the model.
    input_channels : int
        The channels of input of the model.
    mean : tuple of float
        Mean value for normalization of input images.
    std : tuple of float
        Standard deviation for normalization of input images.
    test_bs : int, optional
        The batch size of input to perform the test. The default is 200. 
    prefetch_threads : int, optional
        Number of threads used for input image preprocessing. The default is 4.
    
    Returns
    -------
    data : torch.utils.data.Dataset
        Input image data loaded from a folder or downloaded from the website.

    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ood_loader_dict = dict()
    for data_name in ood_data_list:
        # load OOD test dataset
        ood_data = build_dataset(data_name, mode="test", size=input_size, channels=input_channels,
                                    mean=mean, std=std)
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                                num_workers=prefetch_threads, pin_memory=True)
        ood_loader_dict[data_name] = ood_loader
    return ood_loader_dict
