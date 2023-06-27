import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms

from PIL.Image import Image

from dataclay import DataClayObject, activemethod

from .torch_modules import Net


class MNIST(DataClayObject):
    model: nn.Module
    device: torch.device
    dataset_train: datasets.VisionDataset
    dataset_test: datasets.VisionDataset
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    epoch: int

    @activemethod
    def _train(self, dry_run=False) -> float:
        self.model.train()
        for (data, target) in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if dry_run:
                break
        return loss.item()

    @activemethod
    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        accuracy = 100. * correct / len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), accuracy))
        
        return accuracy
        
    @activemethod
    def _step(self):
        """Update the scheduler and annotate the new epoch"""
        self.scheduler.step()
        self.epoch += 1

    @activemethod
    def prepare(self,
                batch_size: int=64,
                test_batch_size: int=1000,
                lr: float=1.0,
                gamma: float=0.7,
                ):
        self.epoch = 0
        self.device = torch.device("cpu")
        self.model = Net().to(self.device)

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset_train = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset_test = datasets.MNIST('../data', train=False,
                        transform=transform)
        
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)

        self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
        self.test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size)

    def run_epochs(self, n_epochs, dry_run=False):
        for _ in range(n_epochs):
            print("Starting epoch %d" % (self.epoch + 1))
            loss = self._train(dry_run=dry_run)
            print("Train returned loss=%.6f" % loss)
            accuracy = self._test()
            print("Test returned accuracy=%.2f" % accuracy)
            self._step()

    @activemethod
    def inference(self, images: list[Image]) -> tuple[list[int], list[torch.Tensor]]:
        transform=transforms.Compose([
            transforms.Resize((28, 28)),  # MNIST are standard 28x28, but the input may not be
            transforms.functional.invert,
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        labels = list()
        tensors = list()

        for image in images:
            image = transform(image)
            image = image.unsqueeze(0)

            with torch.no_grad():
                output = self.model(image)
                pred = output.argmax(dim=1)
            labels.append(pred.item())
            tensors.append(output)

        return labels, tensors
