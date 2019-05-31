import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from inclearn import factory, utils
from inclearn.models.base import IncrementalLearner
from inclearn.results_utils import get_profile_dict


class ICarl(IncrementalLearner):
    """Implementation of iCarl.

    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

    :param args: An argparse parsed arguments object.
    """
    def __init__(self, args):
        super().__init__()

        self._device = args["device"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._k = args["memory_size"]
        self._n_classes = args["increment"]

        self._features_extractor = factory.get_resnet(args["convnet"], nf=64,
                                                      zero_init_residual=True)
        self._classifier = nn.Linear(self._features_extractor.out_dim, self._n_classes, bias=False)
        torch.nn.init.kaiming_normal_(self._classifier.weight)

        self._examplars = {}
        self._means = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self.to(self._device)

    def forward(self, x):
        x = self._features_extractor(x)
        x = self._classifier(x)
        return x

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        total_start_time = time.time()
        if self._task == 0:
            self._previous_preds = None
            previous_pred_time = 0
            add_n_classes_layer_time = 0
        else:
            print("Computing previous predictions...")
            start_time = time.time()
            self._previous_preds = self._compute_predictions(train_loader)
            if val_loader:
                self._previous_preds_val = self._compute_predictions(val_loader)
            end_time = time.time()
            previous_pred_time = end_time - start_time

            start_time = time.time()
            self._add_n_classes(self._task_size)
            end_time = time.time()
            add_n_classes_layer_time = end_time - start_time

        previous_pred_profile = get_profile_dict(previous_pred_time)
        add_n_classes_layer_profile = get_profile_dict(add_n_classes_layer_time)

        start_time = time.time()
        self._optimizer = factory.get_optimizer(
            self.parameters(),
            self._opt_name,
            self._lr,
            self._weight_decay
        )

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer,
            self._scheduling,
            gamma=self._lr_decay
        )
        end_time = time.time()
        torch_profile = get_profile_dict(end_time - start_time)
        total_end_time = time.time()

        return get_profile_dict(total_end_time - total_start_time, subprofile={
            "previous_pred": previous_pred_profile,
            "add_n_classes_layer": add_n_classes_layer_profile,
            "torch": torch_profile
        })

    def _train_task(self, train_loader, val_loader):
        total_start_time = time.time()
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5, 5))

        print("nb ", len(train_loader.dataset))

        prog_bar = trange(self._n_epochs, desc="Losses.")

        val_loss = 0.

        total_forward_time = 0
        total_loss_time = 0
        total_backward_time = 0
        total_optimizer_time = 0
        total_validation_time = 0

        for epoch in prog_bar:
            _clf_loss, _distil_loss = 0., 0.
            c = 0

            self._scheduler.step()

            for i, ((_, idxes), inputs, targets) in enumerate(train_loader, start=1):
                self._optimizer.zero_grad()

                c += len(idxes)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                targets = utils.to_onehot(targets, self._n_classes).to(self._device)

                # Forward pass
                start_time = time.time()
                logits = self.forward(inputs)
                end_time = time.time()
                total_forward_time += (end_time - start_time)

                # Loss compute
                start_time = time.time()
                clf_loss, distil_loss = self._compute_loss(
                    logits,
                    targets,
                    idxes,
                )
                end_time = time.time()
                total_forward_time += (end_time - start_time)

                if not utils._check_loss(clf_loss) or not utils._check_loss(distil_loss):
                    import pdb; pdb.set_trace()

                loss = clf_loss + distil_loss

                # Backward
                start_time = time.time()
                loss.backward()
                end_time = time.time()
                total_backward_time += (end_time - start_time)

                # Optimizer step
                start_time = time.time()
                self._optimizer.step()
                end_time = time.time()
                total_optimizer_time += (end_time - start_time)

                _clf_loss += clf_loss.item()
                _distil_loss += distil_loss.item()

                if i % 10 == 0 or i >= len(train_loader):
                    prog_bar.set_description(
                    "Clf loss: {}; Distill loss: {}; Val loss: {}".format(
                        round(clf_loss.item(), 3),
                        round(distil_loss.item(), 3),
                        round(val_loss, 3)
                    ))

            start_time = time.time()
            if val_loader is not None:
                val_loss = self._compute_val_loss(val_loader)
            prog_bar.set_description(
                "Clf loss: {}; Distill loss: {}; Val loss: {}".format(
                    round(_clf_loss / c, 3),
                    round(_distil_loss / c, 3),
                    round(val_loss, 2)
            ))
            end_time = time.time()
            total_validation_time += (end_time-start_time)

        total_end_time = time.time()
        forward_profile = get_profile_dict(total_forward_time)
        loss_profile = get_profile_dict(total_loss_time)
        backward_profile = get_profile_dict(total_backward_time)
        optimizer_profile = get_profile_dict(total_optimizer_time)
        validation_profile = get_profile_dict(total_validation_time)
        profile = get_profile_dict(total_end_time - total_start_time, subprofile={
            "forward": forward_profile,
            "loss": loss_profile,
            "backward": backward_profile,
            "optimizer_profile": optimizer_profile,
            "validation_profile": validation_profile,
        })
        profile["epochs"] = self._n_epochs
        profile["train_sample_count"] = len(train_loader.dataset)
        profile["val_sample_count"] = len(val_loader.dataset) if val_loader is not None else 0
        profile["total_samples_processed"] = self._n_epochs * (profile["val_sample_count"] + profile["train_sample_count"])
        return profile

    def _after_task(self, data_loader):
        total_start_time = time.time()

        start_time = time.time()
        self._reduce_examplars()
        end_time = time.time()
        reduce_exemplars_profile = get_profile_dict(end_time-start_time)

        start_time = time.time()
        subprofile = self._build_examplars(data_loader)
        end_time = time.time()
        build_exemplars_profile = get_profile_dict(end_time - start_time, subprofile=subprofile)

        total_end_time = time.time()
        profile = get_profile_dict(time=end_time-start_time, subprofile={
            "reduce_exemplars": reduce_exemplars_profile,
            "build_exemplars": build_exemplars_profile
        })
        return profile

    def _eval_task(self, data_loader):
        total_start_time = time.time()

        ypred, ytrue, classify_profile = self._classify(data_loader)
        assert ypred.shape == ytrue.shape

        total_end_time = time.time()
        profile = get_profile_dict(total_end_time - total_start_time, subprofile={
            "classify": classify_profile
        })
        return ypred, ytrue, profile

    def get_memory_indexes(self):
        return self.examplars

    # -----------
    # Private API
    # -----------

    def _compute_val_loss(self, val_loader):
        total_loss = 0.
        c = 0

        for idx, (idxes, inputs, targets) in enumerate(val_loader, start=1):
            self._optimizer.zero_grad()

            c += len(idxes)

            inputs, targets = inputs.to(self._device), targets.to(self._device)
            targets = utils.to_onehot(targets, self._n_classes).to(self._device)
            logits = self.forward(inputs)

            clf_loss, distil_loss = self._compute_loss(
                logits,
                targets,
                idxes[1],
                train=False
            )

            if not utils._check_loss(clf_loss) or not utils._check_loss(distil_loss):
                import pdb; pdb.set_trace()

            total_loss += (clf_loss + distil_loss).item()

        return total_loss

    def _compute_loss(self, logits, targets, idxes, train=True):
        if self._task == 0:
            # First task, only doing classification loss
            clf_loss = self._clf_loss(logits, targets)
            distil_loss = torch.zeros(1, device=self._device)
        else:
            clf_loss = self._clf_loss(
                logits[..., self._new_task_index:],
                targets[..., self._new_task_index:]
            )

            previous_preds = self._previous_preds if train else self._previous_preds_val
            distil_loss = self._distil_loss(
                logits[..., :self._new_task_index],
                previous_preds[idxes, :self._new_task_index]
            )

        return clf_loss, distil_loss

    def _compute_predictions(self, data_loader):
        preds = torch.zeros(self._n_train_data, self._n_classes, device=self._device)

        for idxes, inputs, _ in data_loader:
            inputs = inputs.to(self._device)
            idxes = idxes[1].to(self._device)

            preds[idxes] = self.forward(inputs).detach()

        return torch.sigmoid(preds)

    def _classify(self, data_loader):
        if self._means is None:
            raise ValueError("Cannot classify without built examplar means,"
                             " Have you forgotten to call `before_task`?")
        if self._means.shape[0] != self._n_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(self._means.shape[0]) +
                " with the number of classes ({}).".format(self._n_classes))

        ypred = []
        ytrue = []

        total_start_time = time.time()
        for _, inputs, targets in data_loader:
            inputs = inputs.to(self._device)

            start_time = time.time()
            features = self._features_extractor(inputs).detach()
            end_time = time.time()
            feature_ext_profile = get_profile_dict(end_time - start_time)

            start_time = time.time()
            preds = self._get_closest(self._means, F.normalize(features))
            end_time = time.time()
            nearestneigh_profile = get_profile_dict(end_time - start_time)

            ypred.extend(preds)
            ytrue.extend(targets)

        total_end_time = time.time()
        subprofile = get_profile_dict(total_end_time - total_start_time, subprofile={
            "feature_extraction": feature_ext_profile,
            "nearest_neighbor": nearestneigh_profile,
        })
        subprofile["classified_samples"] = np.size(ypred,0)
        return np.array(ypred), np.array(ytrue), subprofile

    @property
    def _m(self):
        """Returns the number of examplars per class."""
        return self._k // self._n_classes

    def _add_n_classes(self, n):
        self._n_classes += n

        weights = self._classifier.weight.data
        self._classifier = nn.Linear(
            self._features_extractor.out_dim, self._n_classes,
            bias=False
        ).to(self._device)
        torch.nn.init.kaiming_normal_(self._classifier.weight)

        self._classifier.weight.data[: self._n_classes - n] = weights

        print("Now {} examplars per class.".format(self._m))

    def _extract_features(self, loader):
        features = []
        idxes = []

        task_start_time = time.time()
        total_loader_time = 0
        total_input_xfer_time = 0
        total_feature_extract_time = 0

        last_end_time = time.time()
        for (real_idxes, _), inputs, _ in loader:
            total_loader_time += time.time() - last_end_time

            # Copy to device
            start_time = time.time()
            inputs = inputs.to(self._device)
            end_time = time.time()
            total_input_xfer_time += (end_time-start_time)

            # Extract features
            start_time = time.time()
            features.append(self._features_extractor(inputs).detach())
            end_time = time.time()
            total_feature_extract_time += (end_time - start_time)

            idxes.extend(real_idxes.numpy().tolist())

            last_end_time = time.time()

        features = F.normalize(torch.cat(features), dim=1)
        mean = torch.mean(features, dim=0, keepdim=False)

        task_end_time = time.time()

        profiling = get_profile_dict(time=task_end_time-task_start_time, subprofile={
            "loader_time": total_loader_time,
            "input_xfer": total_input_xfer_time,
            "actual_feature_extraction": total_feature_extract_time
        })
        profiling["total_samples"] = len(loader.dataset)

        return features, mean, idxes, profiling

    @staticmethod
    def _remove_row(matrix, idxes, row_idx):
        new_matrix = torch.cat((matrix[:row_idx, ...], matrix[row_idx + 1:, ...]))
        del matrix
        return new_matrix, idxes[:row_idx] + idxes[row_idx + 1:]

    @staticmethod
    def _get_closest(centers, features):
        pred_labels = []

        features = features
        for feature in features:
            distances = ICarl._dist(centers, feature)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)

    @staticmethod
    def _get_closest_features(center, features):
        distances = ICarl._dist(center, features)
        return distances.argmin().item()

    @staticmethod
    def _dist(a, b):
        return torch.pow(a - b, 2).sum(-1)

    def _build_examplars(self, loader):
        total_start_time = time.time()

        means = []

        lo, hi = 0, self._task * self._task_size
        print("Updating examplars for classes {} -> {}.".format(lo, hi))
        total_updated_exemplars = 0
        start_time = time.time()
        feature_profile = {}
        for class_idx in range(lo, hi):
            loader.dataset.set_idxes(self._examplars[class_idx])
            _, examplar_mean, _, feature_profile = self._extract_features(loader)
            total_updated_exemplars += len(loader.dataset)
            means.append(F.normalize(examplar_mean, dim=0))
        end_time = time.time()
        updating_exemplars_profile = get_profile_dict(end_time-start_time, subprofile={
            "last_class_feature_extraction": feature_profile
        })

        lo, hi = self._task * self._task_size, self._n_classes
        print("Building examplars for classes {} -> {}.".format(lo, hi))

        total_new_exemplars = 0
        total_build_time_start = time.time()
        total_feature_extraction_time = 0
        total_distance_computation_time = 0
        feature_profile = {}
        for class_idx in range(lo, hi):
            examplars_idxes = []

            loader.dataset.set_classes_range(class_idx, class_idx)
            total_new_exemplars += len(loader.dataset)

            start_time = time.time()
            features, class_mean, idxes, subprofile = self._extract_features(loader)
            end_time = time.time()
            total_feature_extraction_time += (end_time-start_time)

            examplars_mean = torch.zeros(self._features_extractor.out_dim, device=self._device)

            class_mean = F.normalize(class_mean, dim=0)

            for i in range(min(self._m, features.shape[0])):
                tmp = F.normalize(
                    (features + examplars_mean) / (i + 1),
                    dim=1
                )
                start_time = time.time()
                distances = self._dist(class_mean, tmp)
                idxes_winner = distances.argsort().cpu().numpy()
                end_time = time.time()
                total_distance_computation_time += (end_time-start_time)

                for idx in idxes_winner:
                    real_idx = idxes[idx]
                    if real_idx in examplars_idxes:
                        continue

                    examplars_idxes.append(real_idx)
                    examplars_mean += features[idx]
                    break

            means.append(F.normalize(examplars_mean / len(examplars_idxes), dim=0))
            self._examplars[class_idx] = examplars_idxes

        self._means = torch.stack(means)

        total_build_time_end = time.time()
        new_exemplars_profile = get_profile_dict(total_build_time_end - total_build_time_start, subprofile={
            "extract_features": get_profile_dict(total_feature_extraction_time, subprofile={"last_class_feature_extraction": subprofile}),
            "distance_compute": get_profile_dict(total_distance_computation_time),
        })

        total_end_time = time.time()

        profile = get_profile_dict(total_end_time-total_start_time, subprofile={
            "updating_exemplars": updating_exemplars_profile,
            "building_exemplars": new_exemplars_profile
        })
        profile["updated_exemplar_count"] = total_updated_exemplars
        profile["new_exemplar_count"] = total_new_exemplars
        return profile

    @property
    def examplars(self):
        return np.array(
            [
                examplar_idx
                for class_examplars in self._examplars.values()
                for examplar_idx in class_examplars
            ]
        )

    def _reduce_examplars(self):
        print("Reducing examplars.")
        for class_idx in range(len(self._examplars)):
            self._examplars[class_idx] = self._examplars[class_idx][: self._m]
