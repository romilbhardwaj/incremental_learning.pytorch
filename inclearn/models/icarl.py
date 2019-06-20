import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.nn import functional as F
from tqdm import tqdm

from inclearn import parser
from inclearn.lib import factory, network, utils
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
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._n_classes = 0

        self._network = network.BasicNet(args["convnet"], device=self._device, use_bias=True)

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._herding_matrix = np.zeros((100, 5000))  # FIXME: nb classes

        self._custom_serialized = ["_network", "_optimizer", "_scheduler"]

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader, new_classes_count):
        self._n_classes += new_classes_count
        self._network.add_classes(new_classes_count)
        print("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

    def _train_task(self, train_loader, val_loader = None, n_epochs = -1):
        print("nb ", len(train_loader.dataset))

        epochs = n_epochs if n_epochs > 0 else self._n_epochs
        for epoch in range(epochs):
            _loss = 0.

            self._scheduler.step()

            prog_bar = tqdm(train_loader)
            c = 0
            for inputs, targets in prog_bar:
                c += 1
                self._optimizer.zero_grad()

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = utils.to_onehot(targets, self._n_classes).to(self._device)
                logits = self._network(inputs)

                loss = self._compute_loss(inputs, logits, targets)

                if not utils._check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                _loss += loss.item()

                prog_bar.set_description(
                    "Task {}/{}, Epoch {}/{} => Clf loss: {}".format(
                        "?", "?",
                        epoch + 1, epochs,
                        round(_loss / c, 3)
                    )
                )
                break

    def _after_task(self, train_loader, flipped_loader):
        self.build_examplars_simple(train_loader, flipped_loader)

        self._old_model = self._network.copy().freeze()

    def _eval_task(self, data_loader):
        ypred, ytrue = compute_accuracy(self._network, data_loader, self._class_means)

        return ypred, ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, logits, targets):
        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            old_targets = torch.sigmoid(self._old_model(inputs).detach())

            new_targets = targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        return loss

    def _compute_predictions(self, data_loader):
        preds = torch.zeros(len(data_loader.dataset), self._n_classes, device=self._device)

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
        if self._means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0]) +
                " with the number of classes ({}).".format(self._n_classes)
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
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(self, inc_dataset):
        self._data_memory, self._targets_memory = [], []
        self._class_means = np.zeros((100, self._network.features_dim))

        for class_idx in range(self._n_classes):
            inputs, loader = inc_dataset.get_class_loader(class_idx, mode="test")
            features, targets = extract_features(
                self._network, loader
            )
            features_flipped, _ = extract_features(
                self._network, inc_dataset.get_class_loader(class_idx, mode="flip")[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                print("Finding examplars for", class_idx)
                self._herding_matrix[class_idx, :] = select_examplars(
                    features, self._memory_per_class
                )

            examplar_mean, alph = compute_examplar_mean(
                features, features_flipped, self._herding_matrix[class_idx], self._memory_per_class
            )
            self._data_memory.append(inputs[np.where(alph == 1)[0]])
            self._targets_memory.append(targets[np.where(alph == 1)[0]])

            self._class_means[class_idx, :] = examplar_mean

        self._data_memory = np.concatenate(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)


    def build_examplars_simple(self, train_loader, flipped_loader):
        self._data_memory, self._targets_memory = [], []
        self._class_means = np.zeros((100, self._network.features_dim))
        inputs = train_loader.dataset.x

        for class_idx in range(self._n_classes):
            features, targets = extract_features(
                self._network, train_loader
            )
            features_flipped, _ = extract_features(
                self._network, flipped_loader
            )

            if class_idx >= self._n_classes - self._task_size:
                print("Finding examplars for", class_idx)
                self._herding_matrix[class_idx, :] = select_examplars(
                    features, self._memory_per_class
                )

            examplar_mean, alph = compute_examplar_mean(
                features, features_flipped, self._herding_matrix[class_idx], self._memory_per_class
            )
            self._data_memory.append(inputs[np.where(alph == 1)[0]])
            self._targets_memory.append(targets[np.where(alph == 1)[0]])

            self._class_means[class_idx, :] = examplar_mean

        self._data_memory = np.concatenate(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)

    def get_memory(self):
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
        self._network.add_classes(self._n_classes)
        self._network.load_state_dict(model_state["_network"])


        self._optimizer = factory.get_optimizer(self._network.parameters(), self._opt_name, self._lr, self._weight_decay)
        self._optimizer.load_state_dict(model_state["_optimizer"])

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, self._scheduling, gamma=self._lr_decay)
        self._scheduler.load_state_dict(model_state["_scheduler"])


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

    print("stats ", np.average(stat_icarl))

    return np.argsort(score_icarl, axis=1)[:, -1], targets_
