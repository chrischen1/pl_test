import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
from torchvision import models

epochs = 60
batch_size = 32
val_batch_size = 32
seed = 42
use_cuda = True
base_lr = 0.001
log_dir = './logs'
checkpoint_format = './checkpoint-{}.pth.tar'
momentum = 0.9
wd = 0.00005


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):

            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), batch_size):
                data_batch = data[i:i + batch_size]
                target_batch = target[i:i + batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                loss.div_(math.ceil(float(len(data)) / batch_size))
                loss.backward()
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = checkpoint_format.format(epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


class TestDataset(Dataset):

    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    num_workers = 1
    allreduce_batch_size = batch_size
    hvd.init()
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(seed)
    resume_from_epoch = 0
    for try_epoch in range(epochs, 0, -1):
        if os.path.exists(checkpoint_format.format(try_epoch)):
            resume_from_epoch = try_epoch
            break

    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    verbose = 1 if hvd.rank() == 0 else 0
    log_writer = SummaryWriter(log_dir) if hvd.rank() == 0 else None
    torch.set_num_threads(num_workers)

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    kwargs['multiprocessing_context'] = 'forkserver'

    train_dataset = TestDataset(torch.rand(10000, 3, 224, 224), torch.randint(1000, (10000,)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs)

    val_dataset = TestDataset(torch.rand(1000, 3, 224, 224), torch.randint(1000, (1000,)))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=val_sampler, **kwargs)
    model = models.resnet50()

    if use_cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=(base_lr * hvd.size()), momentum=momentum, weight_decay=wd)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=hvd.Compression.none,
        backward_passes_per_step=1,
        op=hvd.Average,
        gradient_predivide_factor=1.0)

    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = checkpoint_format.format(resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(resume_from_epoch, epochs):
        train(epoch)
        validate(epoch)
        save_checkpoint(epoch)
