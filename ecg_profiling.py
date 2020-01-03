import torch
import ray
import ray.experimental.serve as serve
from ray.experimental.serve import BackendConfig
import os
from pathlib import Path
import subprocess
from resnet1d import ResNet1D
from server import HTTPActor
import time


class PytorchPredictorECG:
    """
    Data is fired every 8 ms (125Hz).
    But the prediction is made every 30 seconds.
    The data is saved uptill 30 seconds.

    Args:
        model(torch.nn.Module): a pytorch model for prediction.
        cuda(bool): to use_gpu or not.

    """

    def __init__(self, model, cuda=False):
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model = self.model.cuda()

    def __call__(self, flask_request, data):
        if self.cuda:
            data = data.cuda()
        # do the prediction
        result = self.model(data)
        return result.data.cpu().numpy().argmax().item()


# ECG
n_channel = 1
base_filters = 16
kernel_size = 16
n_classes = 2
n_block = 16
model = ResNet1D(in_channels=n_channel,
                 base_filters=base_filters,
                 kernel_size=kernel_size,
                 stride=2,
                 n_block=n_block,
                 groups=base_filters,
                 n_classes=n_classes,
                 downsample_gap=max(n_block//8, 1),
                 increasefilter_gap=max(n_block//4, 1),
                 verbose=False)
print(type(model))

cuda = False
if cuda:
    hw = 'gpu'
else:
    hw = 'cpu'
# initiate serve
p = Path("Resnet1d_base_filters={},kernel_size={},n_block={}"
         "_{}_7600_queries.jsonl".format(base_filters, kernel_size, n_block, hw))
p.touch()
os.environ["HEALTH_PROFILE_PATH"] = str(p.resolve())
serve.init(blocking=True)

# create service


# kwargs_creator = lambda : {'data': 0.}


serve.create_endpoint("ECG")

# create backend
b_config = BackendConfig(num_replicas=1)
serve.create_backend(PytorchPredictorECG, "PredictECG",
                     model, cuda, backend_config=b_config)

# link service and backend
serve.link("ECG", "PredictECG")
handle = serve.get_handle("ECG")
print(handle)
num_queries = 3750
http_actor = HTTPActor.remote(handle, num_queries)
http_actor.run.remote()
# wait for server to start
time.sleep(2)
print("Started client!")
# fire client
procs = []
for _ in range(2):
    ls_output = subprocess.Popen(["go", "run", "ecg_patient.go"])
    procs.append(ls_output)
for p in procs:
    p.wait()
print(serve.stat())
