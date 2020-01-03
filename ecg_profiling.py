import torch
import ray
import ray.experimental.serve as serve
from ray.experimental.serve import BackendConfig
import os
from pathlib import Path
import subprocess
from resnet1d import ResNet1D

class PytorchPredictorECG:
    """
    Data is fired every 8 ms (125Hz).
    But the prediction is made every 30 seconds.
    The data is saved uptill 30 seconds.

    Args:
        model(torch.nn.Module): a pytorch model for prediction.
        cuda(bool): to use_gpu or not.
        frequency_hz(int): Frequency at which data is fired(Hz).
        predict_s(int): time to wait for queries to get accumulated.

    """

    def __init__(self, model, cuda=True,
                 frequency_hz=125, predict_s=30):
        self.cuda = cuda
        self.model = model
        self.frequency_hz = frequency_hz
        self.predict_s = predict_s
        if cuda:
            self.model = self.model.cuda()
        self.size = self.frequency_hz*self.predict_s
        self.historical_data = torch.zeros((1,1,self.size))
        self.num_calls = 0

    def __call__(self, flask_request, data):
        self.historical_data[0][0][self.num_calls] = data
        self.num_calls += 1
        if (self.num_calls == (self.size-1)):
            # make the prediction data
            #predict_tensor = torch.cat(self.historical_data, dim=1)
            #predict_tensor = torch.stack([predict_tensor])
            predict_tensor = self.historical_data
            if self.cuda:
                predict_tensor = predict_tensor.cuda()
            # do the prediction
            result = self.model(predict_tensor)
            # forget the history
            self.historical_data = torch.zeros((1,1,self.size))
            self.num_calls = 0
            return result.data.cpu().numpy().argmax()
        return 1


# ECG
n_channel = 1
base_filters = 64
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
# initiate serve
p = Path("ecg_model_profile.jsonl")
p.touch()
os.environ["SERVE_PROFILE_PATH"] = str(p.resolve())
serve.init(blocking=True)

# create service


kwargs_creator = lambda : {'data': 0.}


serve.create_endpoint("ECG", kwargs_creator=kwargs_creator)

# create backend
b_config = BackendConfig(num_replicas=1)
serve.create_backend(PytorchPredictorECG, "PredictECG",
                     model,False, backend_config=b_config)

# link service and backend
serve.link("ECG", "PredictECG")

# fire client
ls_output = subprocess.Popen(["go", "run", "ecg_patient.go"])
ls_output.communicate()
print(serve.stat())
