import numpy as np
from torchvision import transforms, datasets

from inclearn.lib.CityscapesClassification import CityscapesClassification


def get_transform(mode, dataset_name):
    common_transforms, train_transforms = get_static_transforms(dataset_name)
    if mode == "train":
        trsf = transforms.Compose([*train_transforms, *common_transforms])
    elif mode == "test":
        trsf = transforms.Compose(common_transforms)
    elif mode == "flip":
        trsf = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=1.), *common_transforms]
        )
    else:
        raise NotImplementedError("Unknown mode {}.".format(mode))
    return trsf


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "cityscapes":
        return iCityscapes
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def get_static_transforms(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar100":
        train_transforms = iCIFAR100.train_transforms
        common_transforms = iCIFAR100.common_transforms
    elif dataset_name == "cifar10":
        train_transforms = iCIFAR10.train_transforms
        common_transforms = iCIFAR10.common_transforms
    elif dataset_name == "cityscapes":
        train_transforms = iCityscapes.train_transforms
        common_transforms = iCityscapes.common_transforms
    else:
        raise NotImplementedError("Transform family not supported")

    return common_transforms, train_transforms


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


class iCIFAR100(iCIFAR10):
    base_dataset = datasets.cifar.CIFAR100
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    class_order = [
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]


class iMNIST(DataHandler):
    base_dataset = datasets.MNIST
    train_transforms = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip()]
    common_transforms = [transforms.ToTensor()]


class iPermutedMNIST(iMNIST):

    def _preprocess_initial_data(self, data):
        b, w, h, c = data.shape
        data = data.reshape(b, -1, c)

        permutation = np.random.permutation(w * h)

        data = data[:, permutation, :]

        return data.reshape(b, w, h, c)


class iCityscapes(DataHandler):
    base_dataset = CityscapesClassification
    common_transforms = [transforms.ToTensor(),
                         transforms.Normalize(
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]
                         )
    ]