import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from azureml.core.model import Model

class Net(nn.Module):

    def __init__(self, n_hidden=160):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(28 * 28, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def init():
    global model

    model_path = Model.get_model_path('pytorch_mnist_01')
    model_state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = Net()
    model.load_state_dict(model_state)
    model.eval()

def run(data):
    # クエリ受け取り
    x = torch.tensor(json.loads(data)['x'], dtype=torch.float32).unsqueeze(0)
    # 推論
    with torch.no_grad():
        y = F.softmax(model(x), 1)[0]
        _, index = torch.max(y, 0)

    result = {'label': int(index), 'probability': float(y[index])}
    return result