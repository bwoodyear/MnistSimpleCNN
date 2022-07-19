import os
import argparse
import logging
from collections import OrderedDict
import numpy as np
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchsummary import summary
from ema import EMA
from datasets import MnistDataset
from transforms import RandomRotation
from models.modelM3 import ModelM3
from models.modelM5 import ModelM5
from models.modelM7 import ModelM7
from models.base_model import Model
import ipdb

dirname = os.path.dirname(__file__)
output_path = os.path.join(dirname, '..', 'logs')

wandb.init(project="mnist-baseline-tests", entity="ucl-dark", dir=output_path)


def run(seed=0, epochs=150, kernel_size=5, training_type=None, continual_order=None):
    # random number generator seed ------------------------------------------------#
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        logging.warning("WARNING: CPU will be used for training.")
    else:
        logging.info(f"Training on {torch.cuda.get_device_name()}")

    # data augmentation methods ---------------------------------------------------#
    transform = transforms.Compose([
        RandomRotation(20, seed=seed),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
    ])

    # data loader -----------------------------------------------------------------#

    # # Test just regular dataset
    # regular_dataset = MnistDataset(training=True, transform=transform, regular=True)
    # regular_loader = torch.utils.data.DataLoader(regular_dataset, batch_size=120, shuffle=True)
    # train_loader_dict = {'regular': regular_loader}
    #
    # regular_test_dataset = MnistDataset(training=False, transform=None, regular=True)
    # regular_test_loader = torch.utils.data.DataLoader(regular_test_dataset, batch_size=100, shuffle=False)
    # test_loader_dict = {'regular': regular_test_loader}

    # # Test just fashion dataset
    # fashion_dataset = MnistDataset(training=True, transform=transform, fashion=True)
    # fashion_loader = torch.utils.data.DataLoader(fashion_dataset, batch_size=120, shuffle=True)
    # train_loader_dict = {'fashion': fashion_loader}
    #
    # fashion_test_dataset = MnistDataset(training=False, transform=None, fashion=True)
    # fashion_test_loader = torch.utils.data.DataLoader(fashion_test_dataset, batch_size=100, shuffle=False)
    # test_loader_dict = {'fashion': fashion_test_loader}

    batch_size = 100

    if training_type in {'multi-task', 'multi-task_labels'}:
        train_dataset = MnistDataset(training=True, transform=transform,
                                     regular=True, fashion=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_dict = {'combined': train_loader}
    elif training_type == 'continual':
        regular_dataset = MnistDataset(training=True, transform=transform, regular=True)
        fashion_dataset = MnistDataset(training=True, transform=transform, fashion=True)

        # Create a loader for each of the datasets
        regular_loader = torch.utils.data.DataLoader(regular_dataset, batch_size=batch_size, shuffle=True)
        fashion_loader = torch.utils.data.DataLoader(fashion_dataset, batch_size=batch_size, shuffle=True)

        # Put the data loaders in the specified order
        if continual_order == 'regular_first':
            train_loader_dict = OrderedDict([('regular', regular_loader), ('fashion', fashion_loader)])
        elif continual_order == 'fashion_first':
            train_loader_dict = OrderedDict([('fashion', fashion_loader), ('regular', regular_loader)])
        else:
            raise ValueError(f'Continual learning with this order: {continual_order} not recognised')
    else:
        raise NotImplementedError(f'This training type: {training_type} has not been implemented.')

    regular_test_dataset = MnistDataset(training=False, transform=None, regular=True)
    regular_test_loader = torch.utils.data.DataLoader(regular_test_dataset, batch_size=batch_size, shuffle=False)

    fashion_test_dataset = MnistDataset(training=False, transform=None, fashion=True)
    fashion_test_loader = torch.utils.data.DataLoader(fashion_test_dataset, batch_size=batch_size, shuffle=False)

    test_loader_dict = {'regular': regular_test_loader, 'fashion': fashion_test_loader}

    # model selection -------------------------------------------------------------#

    # model = Model(kernel_size).to(device)
    model = ModelM5().to(device)
    summary(model, (1, 28, 28))

    # hyperparameter selection ----------------------------------------------------#
    learning_rate = 1e-4
    exp_lr_gamma = 0.60

    ema = EMA(model, decay=0.999)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_lr_gamma)

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size,
        "training_type": training_type
    }

    if continual_order:
        wandb.config["continual_order"] = continual_order

    wandb.watch(model, log_freq=100)

    # global variables ------------------------------------------------------------#
    g_step = 0
    # max_correct = 0

    # training and evaluation loop ------------------------------------------------#
    for train_dataset_name, train_loader in train_loader_dict.items():
        for epoch in range(epochs):
            logging.info(f'starting epoch {epoch+1} of {train_dataset_name} MNIST')
            logging.info(f'current learning rate: {lr_scheduler.get_lr()}')

            # --------------------------------------------------------------------------#
            # train process                                                             #
            # --------------------------------------------------------------------------#
            model.train()
            train_loss = 0
            train_corr = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device, dtype=torch.int64)
                optimizer.zero_grad()

                if training_type == 'multi-task_labels' and train_dataset_name == 'regular':
                    output = model(data, 0, device)
                elif training_type == 'multi-task_labels' and train_dataset_name == 'fashion':
                    output = model(data, 1, device)
                else:
                    output = model(data)
                loss = F.nll_loss(output, target)

                train_pred = output.argmax(dim=1, keepdim=True)
                train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
                train_loss += F.nll_loss(output, target, reduction='sum').item()

                loss.backward()
                optimizer.step()
                g_step += 1
                ema(model, g_step)

            train_loss /= len(train_loader.dataset)
            train_accuracy = 100 * train_corr / len(train_loader.dataset)

            wandb.log({f'{train_dataset_name} epoch train loss': train_loss,
                       f'{train_dataset_name} epoch train accuracy': train_accuracy})

            # --------------------------------------------------------------------------#
            # test process                                                              #
            # --------------------------------------------------------------------------#
            model.eval()
            ema.assign(model)

            total_test_loss = 0
            total_correct = 0
            total_pred = np.zeros(0)
            total_target = np.zeros(0)

            # Keep track of loss and correct predictions for each dataset
            dataset_test_loss = {'regular dataset test loss': 0, 'fashion dataset test loss': 0}
            dataset_correct = {'regular dataset test correct': 0, 'fashion dataset test correct': 0}

            with torch.no_grad():
                for test_dataset_name, test_loader in test_loader_dict.items():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device,  dtype=torch.int64)

                        if training_type == 'multi-task_labels' and train_dataset_name == 'regular':
                            output = model(data, 0, device)
                        elif training_type == 'multi-task_labels' and train_dataset_name == 'fashion':
                            output = model(data, 1, device)
                        else:
                            output = model(data)

                        loss = F.nll_loss(output, target, reduction='sum').item()

                        # if np.isnan(loss):
                        #     ipdb.set_trace()

                        total_test_loss += loss
                        dataset_test_loss[f'{test_dataset_name} dataset test loss'] += loss

                        pred = output.argmax(dim=1, keepdim=True)
                        total_pred = np.append(total_pred, pred.cpu().numpy())
                        total_target = np.append(total_target, target.cpu().numpy())
                        
                        batch_correct = pred.eq(target.view_as(pred)).sum().item()
                        total_correct += batch_correct
                        dataset_correct[f'{test_dataset_name} dataset test correct'] += batch_correct

            ema.resume(model)

            # --------------------------------------------------------------------------#
            # output                                                                    #
            # --------------------------------------------------------------------------#
            # test_points = 2e4
            test_points = sum(len(dl.dataset) for dl in test_loader_dict.values())
            total_test_loss /= test_points
            test_accuracy = 100 * total_correct / test_points

            logging.info(f'epoch {epoch+1} test accuracy: {test_accuracy}')

            wandb.log({'epoch test loss': total_test_loss,
                       'epoch test accuracy': test_accuracy})

            wandb.log(dataset_test_loss)

            dataset_accuracy = {k: v/1e5 for k, v in dataset_correct.items()}
            wandb.log(dataset_accuracy)

            # --------------------------------------------------------------------------#
            # update learning rate scheduler                                            #
            # --------------------------------------------------------------------------#
            lr_scheduler.step()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--kernel_size", default=5, type=int)
    p.add_argument("--training_type", required=True, type=str)
    p.add_argument("--continual_order", default='', type=str)
    p.add_argument("--label_level", default=1, type=int)
    p.add_argument("--verbose", default=True, type=bool)
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    run(seed=args.seed,
        epochs=args.epochs,
        kernel_size=args.kernel_size,
        training_type=args.training_type,
        continual_order=args.continual_order)
