import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from azureml.core.run import Run

# 学習結果の取得
run = Run.get_context()

print('PyTorch version: ', torch.__version__)
        
class Net(pl.LightningModule):

    def __init__(self, hparams, num_workers=8):
        super(Net, self).__init__()
        self.hparams = hparams
        self.num_workers = num_workers
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(28 * 28, self.hparams.n_hidden)
        self.fc2 = nn.Linear(self.hparams.n_hidden, 10)
        
    def _dataloader(self, train):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(root=self.hparams.data_dir, train=train, download=True, transform=transform)
        loader = DataLoader(dataset, self.hparams.batch_size, shuffle=True, num_workers=self.num_workers)
        return loader

    def lossfun(self, y, t):
        return F.cross_entropy(y, t)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    @pl.data_loader
    def train_dataloader(self):
        return self._dataloader(train=True)

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results
        
    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader(train=False)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'val_loss': loss, 'val_acc': acc}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        results =  {'val_loss': avg_loss, 'val_acc': avg_acc}
        return results 

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(train=False)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, dest='data_dir', default='./data')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=50)
    parser.add_argument('--epoch', type=int, dest='epoch', default=20)
    parser.add_argument('--lr', type=float, dest='lr', default=0.01)
    parser.add_argument('--n-hidden', type=int, dest='n_hidden', default=100)
    parser.add_argument('--momentum', type=float, dest='momentum', default=0.9)

    hparams = parser.parse_args()
    
    torch.manual_seed(0)
    net = Net(hparams)
    trainer = Trainer(max_nb_epochs=hparams.epoch)

    print('start training')
    trainer.fit(net)
    print('finish training')
    
    # 検証＆評価
    print('Validation Score:', trainer.callback_metrics)
    trainer.test()
    print('Test Score:', trainer.callback_metrics)
    
    # 学習結果の追跡
    run.log('lr', hparams.lr)
    run.log('n_hidden', hparams.n_hidden)
    run.log('momentum', hparams.momentum)
    run.log('accuracy', trainer.callback_metrics)

if __name__ == '__main__':
    main()