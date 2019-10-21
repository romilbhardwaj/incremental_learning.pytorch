import csv
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader

from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CityscapesClassification(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset, converted for classification use.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        sample_list_name (string): Path to the sample_list to use for reading images.
            The list is generated using the generate_sample_list method.
        subsample_idxs (list, optional): Indexes to select from the sample_list.
        use_cache (bool): Cache resized images on disk to speed up data loading.
            WARNING: Might consume a lot of disk.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    COLUMN_ORDER = ["idx", "imgpath", "class", "x0", "y0", "x1", "y1"]

    def __init__(self, root, sample_list_name, subsample_idxs=None, transform=None, target_transform=None,
                 split='train', mode='fine', use_cache=False, resize_res=32):
        super(CityscapesClassification, self).__init__(root)
        self.resize_res = resize_res
        self.use_cache = use_cache
        self.sample_list_name = sample_list_name
        self.sample_list_path = os.path.join(root, "{}_{}_{}.csv".format(sample_list_name, split, mode))
        self.mode = mode
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        self.subsample_idxs = subsample_idxs

        print("Using data cache: {}".format(use_cache))

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        self.samples = pd.read_csv(self.sample_list_path, names=self.COLUMN_ORDER)

        if self.subsample_idxs is not None:
            self.samples = self.samples[self.samples["idx"].isin(self.subsample_idxs)]

    def get_cache_root(self):
        cache_root = os.path.join(self.root, "cache", self.sample_list_name, self.split, str(self.resize_res))
        os.makedirs(cache_root, exist_ok=True)
        return cache_root


    def get_image(self, index):
        '''
        Reads a sample image from disk and applies a crop to get image of an object to be used for classification.
        :param index:
        :return: cropped image
        '''
        sample = self.samples.iloc[index, :]
        img_path = os.path.join(self.root, "leftImg8bit", sample["imgpath"])
        
        if self.use_cache:
            cache_root = self.get_cache_root()
            cache_file_path = os.path.join(cache_root, "sample_{}.pt".format(sample["idx"]))
            if os.path.exists(cache_file_path):
                # Cache hit
                # print("Cache hit on {}".format(cache_file_path))
                return torch.load(cache_file_path)

        image = Image.open(img_path).convert('RGB')
        cropped = image.crop((sample["x0"], sample["y0"], sample["x1"], sample["y1"])).resize((self.resize_res, self.resize_res))
        if self.transform:
            cropped = self.transform(cropped)

        if self.use_cache:
            # If this code executes, must've been a cache miss
            # print("Cache miss on {}".format(cache_file_path))
            torch.save(cropped, cache_file_path)

        return cropped

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = self.get_image(index)

        sample = self.samples.iloc[index, :]
        target = sample['class']
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.samples)

    @property
    def y(self):
        return self.samples["class"].values

    def get_indexes(self, class_filter_list=None):
        '''
        :return: Indexes of elements which belong in the class_filter_list
        '''
        if class_filter_list is None:
            idx_series = self.samples["idx"]
        else:
            idx_series = self.samples[self.samples["class"].isin(class_filter_list)]["idx"]
        return idx_series.values

    def get_targets(self, class_filter_list=None):
        '''
        :return: Targets of elements which belong in the class_filter_list
        '''
        if class_filter_list is None:
            targets = self.samples["class"]
        else:
            targets = self.samples[self.samples["class"].isin(class_filter_list)]["class"]
        return targets.values

    def get_filtered_dataset(self, data_idxs, custom_transforms=None):
        '''
        Subsamples the current dataset to data_idxs and returns a new subsampled CityscapesClassification object.
        :param data_idxs:
        :param custom_transforms:
        :return:
        '''
        trsf = custom_transforms if custom_transforms is not None else self.transform
        return CityscapesClassification(self.root, self.sample_list_name, subsample_idxs=data_idxs,
                                        transform=trsf, target_transform=self.target_transform,
                                        split=self.split, mode=self.mode, use_cache=self.use_cache,
                                        resize_res=self.resize_res)

    def get_filtered_loader(self, data_idxs, custom_transforms=None, **kwargs):
        '''
        Return a Pytorch dataloader with only samples in data_idxs
        :param data_idxs:
        :param custom_transforms:
        :param kwargs:
        :return:
        '''
        subset_dataset = self.get_filtered_dataset(data_idxs, custom_transforms=custom_transforms)
        return DataLoader(
            subset_dataset,
            pin_memory=False,
            **kwargs,
        )

    def get_merged_dataset(self, other_dataset):
        '''
        Merges this instance of a dataset with another dataset and returns a new dataset object.
        :param other_dataset:
        :return:
        '''
        if other_dataset is None:
            return self
        assert isinstance(other_dataset, CityscapesClassification), "The other dataset is not Cityscapes dataset."
        union_idxs = np.union1d(self.samples["idx"].values, other_dataset.samples["idx"].values)
        return CityscapesClassification(self.root, self.sample_list_name, subsample_idxs=union_idxs,
                                        transform=self.transform, target_transform=self.target_transform,
                                        split=self.split, mode=self.mode, use_cache=self.use_cache,
                                        resize_res=self.resize_res)


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
    def generate_sample_list(root, splits=["train", "test", "val"], mode='fine', classes_of_interest_map=None, write_filename=None, train_cities = [], min_res=0):
        '''
        Cityscapes dataset is a segmentation dataset. To convert it into a classification dataset, we must
        get crop out images of the classes we're interested in. This method generates a sample list read by
        a CityscapesClassification object. The sample list is a csv containing the specified classes from train_cities.
        :param root: Dataset root
        :param splits: The splits to generate sample lists for.
        :param mode: Cityscapes mode
        :param classes_of_interest_map: Dictionary mapping str labels to class ids (arbitrary ints)
        :param write_filename: prefix for the sample_list csvs.
        :param train_cities: Cities to use in the training set.
        :param min_res: Minimum required cropped resolution.
        :return: Sample list, and the csv file if write_filename param is provided.
        '''
        result = {}
        mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        for split in splits:
            images_dir = os.path.join(root, 'leftImg8bit', split)
            targets_dir = os.path.join(root, mode, split)
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
                                   ' specified split {} and mode {} are inside the root {} directory'.format(split, mode, root))

            cities = os.listdir(images_dir)
            if split == "train" and train_cities:
                print("Creating train dataset on only {} cities".format(str(train_cities)))
                cities = [c for c in cities if c in train_cities]
            for city in cities:
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
                            if ((x1-x0 > min_res) and (y1-y0 > min_res)):
                                data = [idx_ctr, image, class_id, x0, y0, x1, y1]
                                sample_list.append(data)
                                idx_ctr += 1
                            else:
                                pass
            if write_filename:
                write_mode = 'fine' if mode == 'gtFine' else 'coarse'
                file_path = os.path.join(root, "{}_{}_{}.csv".format(write_filename, split, write_mode))
                CityscapesClassification.dump_to_csv(sample_list, file_path)
            result[split] = sample_list

        return result
