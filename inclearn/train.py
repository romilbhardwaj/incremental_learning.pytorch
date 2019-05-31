import copy
import json
import pprint
import random
import time

import numpy as np
import torch

from inclearn import factory, results_utils, utils
from inclearn.results_utils import get_profile_dict


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    _set_seed(args["seed"])

    factory.set_device(args)

    train_set = factory.get_data(args, train=True)
    test_set = factory.get_data(args, train=False, classes_order=train_set.classes_order)

    train_loader, val_loader = train_set.get_loader(args["validation"])
    test_loader, _ = test_set.get_loader()
    #val_loader = test_loader

    model = factory.get_model(args)

    results = results_utils.get_template_results(args)

    for task in range(0, train_set.total_n_classes // args["increment"]):
        task_start_time = time.time()
        if args["max_task"] == task:
            break

        # Setting current task's classes:
        train_set.set_classes_range(low=task * args["increment"],
                                    high=(task + 1) * args["increment"])
        test_set.set_classes_range(high=(task + 1) * args["increment"])

        model.set_task_info(
            task,
            train_set.total_n_classes,
            args["increment"],
            len(train_set),
            len(test_set)
        )

        # Before Task
        start_time = time.time()
        subprofile = model.before_task(train_loader, val_loader)
        end_time = time.time()
        bt_profile = get_profile_dict(end_time - start_time, subprofile)

        print("train", task * args["increment"], (task + 1) * args["increment"])

        # Train
        start_time = time.time()
        subprofile = model.train_task(train_loader, val_loader)
        end_time = time.time()
        train_profile = get_profile_dict(end_time - start_time, subprofile)


        # After task
        start_time = time.time()
        subprofile = model.after_task(train_loader)
        end_time = time.time()
        at_profile = get_profile_dict(end_time - start_time, subprofile)

        # compute_accuracy task
        start_time = time.time()
        ypred, ytrue, subprofile = model.eval_task(test_loader)
        end_time = time.time()
        eval_profile = get_profile_dict(end_time - start_time, subprofile)
        print("Done with compute accuracy.")

        acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
        print(acc_stats)
        results["results"].append(acc_stats)

        memory_indexes = model.get_memory_indexes()
        train_set.set_memory(memory_indexes)

        # Compute total time
        task_end_time = time.time()
        task_total_time = task_end_time - task_start_time


        results["profile"][task] = get_profile_dict(task_total_time, {
            "before_task": bt_profile,
            "train_task": train_profile,
            "after_task": at_profile,
            "eval_profile": eval_profile,
        })
        print("Profiling for this task: ")
        print(json.dumps(results["profile"][task], sort_keys = True, indent=4))

        profiled_time = sum(k["time"] for k in results["profile"][task]["subprofile"].values())
        print("profiled time: {}, Total time: {}".format(profiled_time, task_total_time))

    if args["name"]:
        results_utils.save_results(results, args["name"])

    del model
    del train_set
    del test_set
    torch.cuda.empty_cache()


def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.
