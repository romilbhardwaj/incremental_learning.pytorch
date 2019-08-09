import numpy as np

# --------
# Datasets
# --------

from inclearn.lib.data_utils import get_transform, _get_dataset


class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        root,
        sample_list_name,
        shuffle=True,
        workers=10,
        batch_size=128,
        increment=10,
        subsample_dataset=1,
        is_sampleincremental=True,
        **kwargs
    ):
        self.sample_list_name = sample_list_name
        self.root = root
        self.subsample_dataset = subsample_dataset
        self.dataset_name = dataset_name
        self.dataset = _get_dataset(self.dataset_name)
        self._setup_data(subsample_dataset=subsample_dataset)
        self.train_transforms = self.dataset.train_transforms
        self.common_transforms = self.dataset.common_transforms

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self._is_sample_incremental = is_sampleincremental
        self._increment_num = increment

        # Sample incremental attributes
        self._taskid_to_idxs_map_train = {}
        self._taskid_to_idxs_map_val = {}
        self._taskid_to_idxs_map_test = {}

    def _setup_data(self, subsample_dataset=1):
        self.train_dataset = self.dataset.base_dataset(self.root, self.sample_list_name, split='train')
        self.test_dataset = self.dataset.base_dataset(self.root, self.sample_list_name, split='test')
        self.val_dataset = self.dataset.base_dataset(self.root, self.sample_list_name, split='val')

        if subsample_dataset < 1:
            indexes = self.train_dataset.get_indexes()
            new_idxes = np.random.choice(indexes, int(len(indexes)*subsample_dataset), replace=False)
            self.train_dataset = self.train_dataset.get_filtered_dataset(new_idxes)

    @property
    def n_tasks(self):
        if self._is_sample_incremental:
            return self._increment_num
        else:
            return len(self.increments)

    def new_task(self, memory=None):
        raise NotImplementedError

    def new_task_incr(self):
        if self._current_task >= self._increment_num:
            raise Exception("No more tasks.")

        if not self._is_sample_incremental:
            raise Exception('Not in sample incremental mode')

        if not self._taskid_to_idxs_map_train:
            print("Initializing indexes for sample incremental")
            # Select uniformly distributed data
            self._taskid_to_idxs_map_train = self.get_idx_order(self.train_dataset, self._increment_num)
            self._taskid_to_idxs_map_val = self.get_idx_order(self.val_dataset, self._increment_num)

            # Use complete test set for each task.
            self._taskid_to_idxs_map_test = {i: self.test_dataset.samples["idx"].values for i in range(0, self._increment_num)}

        train_loader = self.get_custom_index_loader(self._taskid_to_idxs_map_train[self._current_task], split='train', mode='train')
        val_loader = self.get_custom_index_loader(self._taskid_to_idxs_map_val[self._current_task], split='val', mode='test')
        test_loader = self.get_custom_index_loader(self._taskid_to_idxs_map_test[self._current_task], split='test', mode='test')

        unique, counts = np.unique(train_loader.dataset.y, return_counts=True)

        task_info = {
            "min_class": "{}_{}".format(min(unique), counts[np.argmin(unique)]),
            "max_class": "{}_{}".format(max(unique), counts[np.argmin(unique)]),
            "train_idxs": self._taskid_to_idxs_map_train[self._current_task],
            "test_idxs": self._taskid_to_idxs_map_test[self._current_task],
            "val_idxs": self._taskid_to_idxs_map_val[self._current_task],
            "task": self._current_task,
            "max_task": self._increment_num,
            "n_train_data": len(train_loader),
            "n_test_data": len(test_loader)
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    @staticmethod
    def get_idx_order(dataset, num_tasks):
        # Generates the idx ordering for sample incremental task
        # Use this to populate the _taskid_to_idxs_maps to contain the idxs to be provided for each task increment
        # Selects N/num_tasks samples per class for k classes, where N is the number of samples per class
        classid_to_idx_map = {}
        samples_df = dataset.samples
        y = samples_df["class"].values
        for class_id in np.unique(y):
            classid_to_idx_map[class_id] = samples_df[samples_df["class"] == class_id]["idx"].values

        idx_order = {}
        for t in range(0, num_tasks):
            idx_order[t] = []  # temporary list, later concat into single vector
            for class_id, class_idxs in classid_to_idx_map.items():
                samples_per_task = int(len(class_idxs) / num_tasks)
                idx_order[t].append(class_idxs[t * samples_per_task:(t + 1) * samples_per_task])

            if idx_order[t]:  # If any elements in the list, concat into numpy arr
                idx_order[t] = np.concatenate(idx_order[t])

        return idx_order


    def get_custom_loader(self, class_indexes, split="train", mode="test", shuffle=True):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if split == "train":
            ds = self.train_dataset
        elif split == "test":
            ds = self.test_dataset
        elif split == "val":
            ds = self.val_dataset
        else:
            raise NotImplementedError("Unknown split {}".format(split))

        class_data_indexes = ds.filter_class(class_indexes)
        return self.get_custom_index_loader(class_data_indexes, split, mode, shuffle)

    def get_custom_index_loader(self, data_indexes=None, split="train", mode="test", shuffle=True):
        """Returns a custom loader.

        :param data_indexes: A list of data indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if split == "train":
            ds = self.train_dataset
        elif split == "test":
            ds = self.test_dataset
        elif split == "val":
            ds = self.val_dataset
        else:
            raise NotImplementedError("Unknown split {}".format(split))

        trsf = get_transform(mode, self.dataset_name)
        return ds.get_filtered_loader(
            data_indexes,
            custom_transforms=trsf,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._workers
        )

# --------------
# Data utilities
# --------------