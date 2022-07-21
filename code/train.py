import os
import argparse
import wandb
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
from datasets import MnistDataset
# from models.modelM3 import ModelM3
# from models.modelM5 import ModelM5
# from models.modelM7 import ModelM7
from models.base_model import Model
from collections import OrderedDict
import ipdb

dirname = os.path.dirname(__file__)
output_path = os.path.join(dirname, '..', 'logs')


def run(seed=0, epochs=None, lr=None, kernel_size=None, training_type=None, continual_order=None,
        norm=None, reg_lambda=None, label_level=None):

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
        transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomAffine(0, translate=(0.2, 0.2), interpolation=transforms.InterpolationMode.NEAREST)
    ])

    # data loader -----------------------------------------------------------------#
    batch_size = 100

    if training_type in {'multi-task', 'multi-task_labels'}:
        train_dataset = MnistDataset(training=True, transform=transform,
                                     digit=True, fashion=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_dict = {'combined': train_loader}
    elif training_type == 'continual':
        digit_dataset = MnistDataset(training=True, transform=transform, digit=True)
        fashion_dataset = MnistDataset(training=True, transform=transform, fashion=True)

        # Create a loader for each of the datasets
        digit_loader = torch.utils.data.DataLoader(digit_dataset, batch_size=batch_size, shuffle=True)
        fashion_loader = torch.utils.data.DataLoader(fashion_dataset, batch_size=batch_size, shuffle=True)

        # Put the data loaders in the specified order
        if continual_order == 'digit_first':
            train_loader_dict = OrderedDict([('digit', digit_loader), ('fashion', fashion_loader)])
        elif continual_order == 'fashion_first':
            train_loader_dict = OrderedDict([('fashion', fashion_loader), ('digit', digit_loader)])
        else:
            raise ValueError(f'Continual learning with this order: {continual_order} not recognised')
    else:
        raise NotImplementedError(f'This training type: {training_type} has not been implemented.')

    digit_test_dataset = MnistDataset(training=False, transform=None, digit=True)
    digit_test_loader = torch.utils.data.DataLoader(digit_test_dataset, batch_size=batch_size, shuffle=False)

    fashion_test_dataset = MnistDataset(training=False, transform=None, fashion=True)
    fashion_test_loader = torch.utils.data.DataLoader(fashion_test_dataset, batch_size=batch_size, shuffle=False)

    test_loader_dict = {'digit': digit_test_loader, 'fashion': fashion_test_loader}

    # model selection -------------------------------------------------------------#
    if 'labels' in training_type:
        model = Model(kernel_size=kernel_size, label_level=label_level).to(device)
    else:
        model = Model().to(device)

    # To get an overview of the model size and memory requirements feed dummy values into summary
    dummy_data = torch.rand((batch_size, 1, 28, 28))
    dummy_labels = torch.randint(0, 2, (batch_size,))
    summary(model, input_data=[dummy_data, dummy_labels])

    # hyperparameter selection ----------------------------------------------------#
    exp_lr_gamma = 0.95

    if norm == 'l2':
        # L2 is the default weight decay for Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_lambda)
    else:
        # Otherwise use standard Adam
        optimizer = optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_lr_gamma)

    # Setup wandb logging
    wandb.init(project="mnist-baseline-tests", entity="ucl-dark", dir=output_path, reinit=True,
               config={
                   "learning_rate": lr,
                   "epochs": epochs,
                   "seed": seed,
                   "batch_size": batch_size,
                   "training_type": training_type
               })

    if continual_order:
        wandb.config.update({"continual_order": continual_order})

    wandb.watch(model, log_freq=100)

    # training and evaluation loop ------------------------------------------------#
    for train_dataset_name, train_loader in train_loader_dict.items():
        for epoch in range(epochs):
            logging.info(f'starting epoch {epoch+1} of {train_dataset_name} MNIST')
            logging.info(f'current learning rate: {lr_scheduler.get_last_lr()[0]:.1E}')

            # --------------------------------------------------------------------------#
            # train process                                                             #
            # --------------------------------------------------------------------------#
            model.train()
            train_loss = 0
            train_corr = 0
            for batch_idx, (data, target_and_label) in enumerate(train_loader):

                data = data.to(device)
                # Split the targets and labels
                target = torch.flatten(target_and_label[:, 0]).to(device, dtype=torch.int64)
                labels = torch.flatten(target_and_label[:, 1]).to(device, dtype=torch.int64)

                optimizer.zero_grad()

                if 'labels' in training_type:
                    output = model(data, labels)
                else:
                    output = model(data)

                if norm == 'l1':
                    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                    loss = F.nll_loss(output, target) + reg_lambda * l1_loss
                else:
                    loss = F.nll_loss(output, target)

                train_pred = output.argmax(dim=1, keepdim=True)
                train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
                train_loss += F.nll_loss(output, target, reduction='sum').item()

                loss.backward()
                optimizer.step()

            train_loss /= len(train_loader.dataset)
            train_accuracy = 100 * train_corr / len(train_loader.dataset)

            wandb.log({f'{train_dataset_name} epoch train loss': train_loss,
                       f'{train_dataset_name} epoch train accuracy': train_accuracy})

            # --------------------------------------------------------------------------#
            # test process                                                              #
            # --------------------------------------------------------------------------#
            model.eval()

            total_test_loss = 0
            total_correct = 0
            total_pred = np.zeros(0)
            total_target = np.zeros(0)

            # Keep track of loss and correct predictions for each dataset
            dataset_test_loss = {'digit': 0, 'fashion': 0}
            dataset_test_correct = {'digit': 0, 'fashion': 0}

            with torch.no_grad():
                for test_dataset_name, test_loader in test_loader_dict.items():
                    for data, target_and_label in test_loader:

                        data = data.to(device)
                        # Split the targets and labels
                        target = torch.flatten(target_and_label[:, 0]).to(device, dtype=torch.int64)
                        labels = torch.flatten(target_and_label[:, 1]).to(device, dtype=torch.int64)

                        if 'labels' in training_type:
                            output = model(data, labels)
                        else:
                            output = model(data)

                        loss = F.nll_loss(output, target, reduction='sum').item()

                        total_test_loss += loss
                        dataset_test_loss[test_dataset_name] += loss

                        pred = output.argmax(dim=1, keepdim=True)
                        total_pred = np.append(total_pred, pred.cpu().numpy())
                        total_target = np.append(total_target, target.cpu().numpy())

                        batch_correct = pred.eq(target.view_as(pred)).sum().item()
                        total_correct += batch_correct
                        dataset_test_correct[test_dataset_name] += batch_correct

            # --------------------------------------------------------------------------#
            # output                                                                    #
            # --------------------------------------------------------------------------#
            # Find the number of test points for each dataset
            test_points = {k: len(dl.dataset) for k, dl in test_loader_dict.items()}
            total_test_points = sum(test_points.values())

            total_test_loss /= total_test_points
            test_accuracy = 100 * total_correct / total_test_points

            logging.info(f'epoch {epoch+1} test accuracy: {test_accuracy}')

            wandb.log({'epoch test loss': total_test_loss,
                       'epoch test accuracy': test_accuracy})

            wandb.log({f'{k} dataset test loss': v for k, v in dataset_test_loss.items()})

            dataset_test_accuracy = {f'{k} dataset test accuracy': 100*v/test_points[k]
                                     for k, v in dataset_test_correct.items()}
            wandb.log(dataset_test_accuracy)

            # --------------------------------------------------------------------------#
            # update learning rate scheduler                                            #
            # --------------------------------------------------------------------------#
            lr_scheduler.step()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--seeds", type=int, required=True, nargs='*', help='random seeds for torch')
    p.add_argument("--lr", default=1e-3, type=int, nargs='*', help='random seeds for torch')
    p.add_argument("--epochs", default=10, type=int, help='number of epochs to train for')
    p.add_argument("--kernel_size", default=5, type=int, help='size of convolution kernels to use')
    p.add_argument("-t", "--training_type", required=True, type=str, help='type of training for the datasets',
                   choices=['multi-task', 'continual', 'multi-task_labels', 'continual_labels'])
    p.add_argument("--continual_order", default='', type=str, help='dataset order for continual training',
                   choices=['digit_first', 'fashion_first'])
    p.add_argument("--label_level", type=int, help='which number layer to insert dataset labels at in the network')
    p.add_argument("--norm", type=str, help='type of norm for regularisation',
                   choices=['l1', 'l2'])
    p.add_argument("--reg_lambda", type=float, default=1e-3, help='lambda for the regularization term')
    p.add_argument("-v", "--verbose", action='store_true')
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    for s in args.seeds:
        logging.info(f'\n\nStaring run with seed {s}\n')
        run(seed=s,
            epochs=args.epochs,
            lr=args.lr,
            kernel_size=args.kernel_size,
            training_type=args.training_type,
            continual_order=args.continual_order,
            label_level=args.label_level)
