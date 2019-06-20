from inclearn import parser
from inclearn.lib import factory, utils
from inclearn.models.icarl import ICarl

args = parser.get_parser().parse_args()
args = vars(args)  # Converting argparse Namespace to a dict.
args["device"] = "cpu"

print("Restoring")
model = ICarl.from_checkpoint("m.pt")
print("Restoring done")

inc_dataset = factory.get_data(args)
memory = None
task_info, train_loader, test_loader = inc_dataset.new_task(memory)

model.eval_task(train_loader)