from inclearn import parser
from inclearn.lib import factory, utils
from inclearn.models.icarl import ICarl

args = parser.get_parser().parse_args()
args = vars(args)  # Converting argparse Namespace to a dict.
args["device"] = "cpu"

inc_dataset = factory.get_data(args)
args["classes_order"] = inc_dataset.class_order
model = factory.get_model(args)

memory = None
task_info, train_loader, test_loader = inc_dataset.new_task(memory)

model.set_task_info(
    task=task_info["task"],
    total_n_classes=task_info["max_class"],
    increment=task_info["increment"],
    n_train_data=task_info["n_train_data"],
    n_test_data=task_info["n_test_data"],
    n_tasks=task_info["max_task"]
)

model.eval()
model.before_task(train_loader, None)
print("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
model._n_epochs = 1
model.train()
model.train_task(train_loader, None)
model.eval()
model.after_task(inc_dataset)

print("Eval on {}->{}.".format(0, task_info["max_class"]))
ypred, ytrue = model.eval_task(test_loader)
acc_stats = utils.compute_accuracy(ypred, ytrue, task_size=args["increment"])
print(acc_stats)

print("Checkpointing!")
model.checkpoint("m.pt")
print("Checkpointing done!")