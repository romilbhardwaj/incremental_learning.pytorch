import abc
import logging

LOGGER = logging.Logger("IncLearn", level="INFO")


class IncrementalLearner(abc.ABC):
    """Base incremental learner.

    Methods are called in this order (& repeated for each new task):

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task
    """

    def __init__(self, *args, **kwargs):
        pass

    def set_task_info(self, task, total_n_classes, increment, n_train_data, n_test_data,
                      n_tasks):
        self._task = task
        self._task_size = increment
        self._total_n_classes = total_n_classes
        self._n_train_data = n_train_data
        self._n_test_data = n_test_data
        self._n_tasks = n_tasks

    def before_task(self, train_loader, val_loader, **kwargs):
        LOGGER.info("Before task")
        self.eval()
        self._before_task(train_loader, val_loader, **kwargs)

    def train_task(self, train_loader, val_loader, n_epochs=-1):
        LOGGER.info("train task")
        self.train()
        self._train_task(train_loader, val_loader, n_epochs)

    def after_task(self, *args, **kwargs):
        LOGGER.info("after task")
        self.eval()
        self._after_task(*args, **kwargs)

    def eval_task(self, data_loader):
        LOGGER.info("eval task")
        self.eval()
        return self._eval_task(data_loader)

    def get_memory(self):
        return None

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def checkpoint(self, checkpoint_path):
        raise NotImplementedError

    def restore(self, checkpoint_path):
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        model = cls.__new__(cls)
        model.restore(checkpoint_path)
        return model

    def _before_task(self, data_loader):
        pass

    def _train_task(self, train_loader, val_loader, n_epochs):
        raise NotImplementedError

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, data_loader):
        raise NotImplementedError

    @property
    def _new_task_index(self):
        return self._task * self._task_size
