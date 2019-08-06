import csv
import json
import os
import pandas as pd

from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CityscapesClassification(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset, converted for classification use.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """


    def __init__(self, root, sample_list_path, transform=None, target_transform=None):
        super(CityscapesClassification, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        self.samples = pd.read_csv(sample_list_path, names=["idx", "imgpath", "class", "x0", "y0", "x1", "y1"])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        sample = self.samples.iloc[index, :]
        image = Image.open(os.path.join(self.root, "leftImg8bit", sample["imgpath"])).convert('RGB')
        cropped = image.crop((sample["x0"], sample["y0"], sample["x1"], sample["y1"]))
        target = sample['class']

        if self.transform:
            cropped = self.transform(cropped)

        if self.target_transform:
            target = self.target_transform(target)

        return cropped, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_json(path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def _get_target_suffix(mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

    @staticmethod
    def get_rectangle_from_polygon(polygon_coords):
        x_list = [c[0] for c in polygon_coords]
        y_list = [c[1] for c in polygon_coords]
        return [min(x_list), min(y_list), max(x_list), max(y_list)]

    @staticmethod
    def dump_to_csv(list_of_lists, filepath):
        with open(filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerows(list_of_lists)

    @staticmethod
    def generate_sample_list(root, split='train', mode='fine', classes_of_interest_map=None, write_filename=None):
        root = root
        mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        images_dir = os.path.join(root, 'leftImg8bit', split)
        targets_dir = os.path.join(root, mode, split)
        split = split
        if classes_of_interest_map is None:
            # This is a map mapping str labels to class ids
            classes_of_interest_map = {'person': 0,
                                       'car': 1,
                                       'truck': 2,
                                       'bus': 3,
                                       'bicycle': 4,
                                       'motorcycle': 5
                                       }

        sample_list = []  # A list of list of format [idx, imgpath, x0, y0, x1, y1]
        idx_ctr = 0

        if mode not in ['gtFine', 'gtCoarse']:
            raise ValueError('Invalid mode! Please use mode="fine" or mode="coarse"')

        if mode == 'fine' and split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode "fine"! Please use split="train", split="test"'
                             ' or split="val"')
        elif mode == 'coarse' and split not in ['train', 'train_extra', 'val']:
            raise ValueError('Invalid split for mode "coarse"! Please use split="train", split="train_extra"'
                             ' or split="val"')

        if not os.path.isdir(images_dir) or not os.path.isdir(targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        for city in os.listdir(images_dir):
            img_dir = os.path.join(images_dir, city)
            target_dir = os.path.join(targets_dir, city)
            for file_name in os.listdir(img_dir):
                image = os.path.join(split, city, file_name)

                target_filename = '{}_{}_polygons.json'.format(file_name.split('_leftImg8bit')[0], mode)
                target_data = CityscapesClassification._load_json(os.path.join(target_dir, target_filename))

                # Iterate over objects and get objs of interest
                for obj in target_data['objects']:
                    if obj["label"] in classes_of_interest_map.keys():
                        class_id = classes_of_interest_map[obj["label"]]
                        x0, y0, x1, y1 = CityscapesClassification.get_rectangle_from_polygon(obj["polygon"])
                        data = [idx_ctr, image, class_id, x0, y0, x1, y1]
                        sample_list.append(data)
                        idx_ctr += 1

        if write_filename:
            file_path = os.path.join(root, "{}_{}_{}.csv".format(write_filename, split, mode))
            CityscapesClassification.dump_to_csv(sample_list, file_path)

        return sample_list
