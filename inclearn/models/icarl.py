import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm

from inclearn import parser
from inclearn.lib import factory, network, utils
from inclearn.lib.data import DummyDataset
from inclearn.models.base import IncrementalLearner

EPSILON = 1e-8


class ICarl(IncrementalLearner):
    """Implementation of iCarl.

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """

    def __init__(self, args = None):
        super().__init__()

        if args is None:
            args = {}

        args = parser.fill_in_defaults(args)

        self._args = args
        # factory.set_device(args)
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #args["device"]
        print("Using device: {}".format(str(self._device)))
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]


        # Sample incremental stuff
        self._new_classes = set()
        self._trained_classes = set()
        self._samples_trained_per_class = {}    # A map storing the number of samples the model has been trained on per class
        self._label_to_convclass_map = {}   # Maps the real labels to the convnet class

        self._network = network.BasicNet(args["convnet"], device=self._device, use_bias=True)

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._herding_matrix = {}
        self._exemplar_memory = {}

        self._custom_serialized = ["_network", "_optimizer", "_scheduler"]

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    # ----------
    # Public API
    # ----------

    @property
    def _new_classes_count(self):
        return len(self._new_classes)

    @property
    def _trained_classes_count(self):
        return len(self._trained_classes)

    @property
    def _total_classes_count(self):
        return self._new_classes_count + self._trained_classes_count

    def _before_task(self, train_loader, new_classes_count = None):
        trainloader_classes = set(train_loader.dataset.y)
        self._new_classes = trainloader_classes - self._trained_classes
        # Update label map for new classes
        for class_label in self._new_classes:
            self._label_to_convclass_map[class_label] = len(self._label_to_convclass_map)

        self._network.add_classes(self._new_classes_count)
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            filter(lambda p: p.requires_grad, self._network.parameters()), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

    def _train_task(self, *args, n_epochs=1, **kwargs):
        epochs = n_epochs if n_epochs > 0 else self._n_epochs
        result = None
        for epoch in range(epochs):
            print("Epoch: {}/{}".format(epoch, epochs))
            result = self._train_epoch(*args, **kwargs)
        return result

    def _generate_class_weights(self, train_loader):
        class_ids, counts = np.unique(train_loader.dataset.y, return_counts=True)
        new_sample_counts = dict(zip(class_ids, counts))
        distillation_class_weights = {}
        for c in class_ids:
            distillation_class_weights[c] = self.get_old_sample_weight(self._samples_trained_per_class.get(c, 0), new_sample_counts.get(c, 0))
        self.distillation_class_weights = distillation_class_weights
        print(self.distillation_class_weights)


    def _train_epoch(self, train_loader, val_loader = None):
        _loss, val_loss = 0., 0.

        self._scheduler.step()

        self.current_train_loader = train_loader
        self._generate_class_weights(train_loader)
        prog_bar = tqdm(train_loader)
        for i, (inputs, targets) in enumerate(prog_bar, start=1):
            self._optimizer.zero_grad()

            loss = self._forward_loss(inputs, targets)

            if not utils._check_loss(loss):
                import pdb
                pdb.set_trace()

            loss.backward()
            self._optimizer.step()

            _loss += loss.item()

            if val_loader is not None and i == len(train_loader):
                for inputs, targets in val_loader:
                    val_loss += self._forward_loss(inputs, targets).item()

            prog_bar.set_description(
                "Task {}/{}, Epoch {}/{} => Clf loss: {}, Val loss: {}".format(
                    "?", "?",
                    "?", "?",
                    round(_loss / i, 3),
                    round(val_loss, 3)
                )
            )
        return _loss / i, val_loss

    def _forward_loss(self, inputs, targets):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        targets = utils.to_onehot(targets, self._total_classes_count, label_map=self._label_to_convclass_map).to(self._device)
        logits = self._network(inputs)

        return self._compute_loss(inputs, logits, targets)

    def _after_task(self, train_loader):
        self.build_examplars(train_loader)

        self._old_model = self._network.copy().freeze()

        # Update the local state with new_sample_counts
        self._samples_trained_per_class = self.get_new_total_class_counts(train_loader, self._samples_trained_per_class)

        self._trained_classes = self._trained_classes.union(self._new_classes)
        self._new_classes = set()

    def _eval_task(self, data_loader):
        ypred, ytrue = compute_accuracy(self._network, data_loader, self._class_means)


        return ypred, ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, logits, targets):
        if self._old_model is None:
            # print("No old model found for loss computing, using direct loss")
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            old_targets = torch.sigmoid(self._old_model(inputs).detach())
            new_targets = targets.clone()

            _, class_tensor = targets.max(1)
            weight_list = [self.distillation_class_weights[c.item()] for c in class_tensor]
            old_sample_weight_tensor = self.iterable_to_tensor(weight_list).unsqueeze(1)

            if self._new_classes_count > 0:
                # Weighted average for classes already seen
                new_targets[..., :-self._new_classes_count] = old_sample_weight_tensor * old_targets + (1 - old_sample_weight_tensor) * targets[..., :-self._new_classes_count]
            else:
                new_targets[..., :] = old_sample_weight_tensor * old_targets + (1 - old_sample_weight_tensor) * targets[..., :]

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        return loss

    def _compute_predictions(self, data_loader):
        preds = torch.zeros(len(data_loader.dataset), self.total_classes_count, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            preds[idxes] = self._network(inputs).detach()

        return torch.sigmoid(preds)

    def _classify(self, data_loader):
        if self._means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
                " Have you forgotten to call `before_task`?"
            )
        if self._means.shape[0] != self.total_classes_count:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0]) +
                " with the number of classes ({}).".format(self.total_classes_count)
            )

        ypred = []
        ytrue = []

        for _, inputs, targets in data_loader:
            inputs = inputs.to(self._device)

            features = self._network.extract(inputs).detach()
            preds = self._get_closest(self._means, F.normalize(features))

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        return self._memory_size // self._total_classes_count

    # -----------------
    # Memory management
    # -----------------

    def get_merged_dataset(self, class_exemplars, current_dataset):
        if class_exemplars is None:
            x = current_dataset.x
            y = current_dataset.y
        else:
            exemplar_x, exemplar_y = class_exemplars
            x = np.concatenate((current_dataset.x, exemplar_x))
            y = np.concatenate((current_dataset.y, exemplar_y))
        # print("Merged shape: {}".format(x.shape))
        return DummyDataset(x,y, trsf=current_dataset.trsf)

    def build_examplars(self, train_loader):
        print("Building & updating memory.")

        self._data_memory, self._targets_memory = [], []
        self._class_means = np.zeros((100, self._network.features_dim)) # We must reconstruct the means at every iteration
        current_dataset = train_loader.dataset

        for class_idx in self._trained_classes.union(self._new_classes):
            class_merged_dataset = self.get_merged_dataset(self._exemplar_memory.get(class_idx, None), current_dataset)
            inputs, loader = class_merged_dataset.get_custom_loader(class_idx, mode="test")
            input_classes = loader.dataset.y
            print("Class {}, Loader size: {}".format(class_idx, len(loader.dataset)))
            features, targets = extract_features(
                self._network, loader
            )
            features_flipped, _ = extract_features(
                self._network, class_merged_dataset.get_custom_loader(class_idx, mode="flip")[1]
            )

            # if class_idx in self._exemplar_features_memory:
            #     print("Found memory.")
            #     features = np.concatenate(features, self._exemplar_features_memory[class_idx])
            #     features_flipped = np.concatenate(features_flipped, self._exemplar_features_memory_flipped[class_idx])

            # if class_idx not in self._trained_classes:  # TODO: Run this for all data
            self._herding_matrix[class_idx] = select_examplars(
                features, self._memory_per_class
                )

            examplar_mean, alph = compute_examplar_mean(
                features, features_flipped, self._herding_matrix[class_idx], self._memory_per_class
            )



            #TODO: Check the size of alph and the count of alph==1
            # self._data_memory.append(inputs[np.where(alph == 1)[0]])
            # self._targets_memory.append(targets[np.where(alph == 1)[0]])

            self._exemplar_memory[class_idx] = (inputs[np.where(alph == 1)[0]], input_classes[np.where(alph == 1)[0]])


            self._class_means[class_idx, :] = examplar_mean

        # self._data_memory = np.concatenate(self._data_memory)
        # self._targets_memory = np.concatenate(self._targets_memory)
        print(self._class_means)

    def get_memory(self):
        raise NotImplementedError("Disabled because memory no longer stored in build_exemplars")
        return self._data_memory, self._targets_memory

    def checkpoint(self, checkpoint_path):
        model_state = {}

        # Save state_dicts for pytorch objects
        model_state["_network"] = self._network.state_dict()
        model_state["_optimizer"] = self._optimizer.state_dict()
        model_state["_scheduler"] = self._scheduler.state_dict()

        # Save everything else
        for key, value in vars(self).items():
            if key not in self._custom_serialized:
                model_state[key] = value
        torch.save(model_state, checkpoint_path)

    def restore(self, checkpoint_path):
        model_state = torch.load(checkpoint_path)

        self._args = model_state["_args"]

        for key, value in model_state.items():
            if key not in model_state["_custom_serialized"]:
                self.__setattr__(key, value)

        self._network = network.BasicNet(self._args["convnet"], device=self._device, use_bias=True)
        self._network.add_classes(self.total_classes_count)
        self._network.load_state_dict(model_state["_network"])


        self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()), self._opt_name, self._lr, self._weight_decay)
        self._optimizer.load_state_dict(model_state["_optimizer"])

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, self._scheduling, gamma=self._lr_decay)
        self._scheduler.load_state_dict(model_state["_scheduler"])

    def freeze_layers(self, layers_to_freeze = 7):
        self._network.freeze_conv(layers_to_freeze)
        # update optimizer
        self._optimizer = factory.get_optimizer(
            filter(lambda p: p.requires_grad, self._network.parameters()), self._opt_name, self._lr, self._weight_decay
        )


    @staticmethod
    def get_new_total_class_counts(train_loader, samples_trained_dict):
        # Get class counts in the train loader
        class_ids, counts = np.unique(train_loader.dataset.y, return_counts=True)
        new_sample_counts = dict(zip(class_ids, counts))

        # Update the local state with new_sample_counts
        return {key: samples_trained_dict.get(key, 0) + new_sample_counts.get(key, 0)
                                           for key in set(samples_trained_dict) | set(new_sample_counts)}

    @staticmethod
    def get_old_sample_weight(num_old_samples, num_new_samples):
        def func(x):
            return 1 - (1 / (1 + (1 / x)))

        if num_old_samples == 0:
            return 1  # Maximize distillation when creating new classes
        elif num_new_samples == 0:
            raise ValueError("num_new_samples is 0, this does not make sense. Why are you calling this method?")
        else:
            ratio = num_new_samples / num_old_samples
            return func(ratio)

    def iterable_to_tensor(self, l):
        return torch.from_numpy(np.array(list(l))).float().to(self._device)

def extract_features(model, loader):
    targets, features = [], []

    for _inputs, _targets in loader:
        _targets = _targets.numpy()
        _features = model.extract(_inputs.to(model.device)).detach().cpu().numpy()

        features.append(_features)
        targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)


def select_examplars(features, nb_max):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_max, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    return herding_matrix


def compute_examplar_mean(feat_norm, feat_flip, herding_mat, nb_max):
    D = feat_norm.T
    D = D / (np.linalg.norm(D, axis=0) + EPSILON)

    D2 = feat_flip.T
    D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

    alph = herding_mat
    alph = (alph > 0) * (alph < nb_max + 1) * 1.

    alph_mean = alph / np.sum(alph)

    mean = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
    mean /= np.linalg.norm(mean)

    return mean, alph


def compute_accuracy(model, loader, class_means):
    features, targets_ = extract_features(model, loader)

    targets = np.zeros((targets_.shape[0], 100), np.float32)
    targets[range(len(targets_)), targets_.astype('int32')] = 1.
    features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

    # Compute score for iCaRL
    sqd = cdist(class_means, features, 'sqeuclidean')
    score_icarl = (-sqd).T

    # Compute the accuracy over the batch
    stat_icarl = [
        ll in best
        for ll, best in zip(targets_.astype('int32'),
                            np.argsort(score_icarl, axis=1)[:, -1:])
    ]
    #print("stats ", np.average(stat_icarl))

    return np.argsort(score_icarl, axis=1)[:, -1], targets_
