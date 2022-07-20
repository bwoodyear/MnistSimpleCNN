## MnistSimpleCNN

This repository is fork of "An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition".
This was chosen as it is the top performing result on the papers with code leaderboard for MNIST
(https://paperswithcode.com/sota/image-classification-on-mnist) and the individual models are simple and so are 
straightforward to both train and change.

Paper url is https://arxiv.org/abs/2008.10400.

These models are used as the foundation for our tests of sparsity methods. In order to generate some simple
baselines MNIST and Fashion-MNIST are used as the two tasks. A third dataset (potentially 
https://github.com/rois-codh/kmnist) will be added too.

### Running 

```bash
python3 train.py --training_type multi-task --epochs 20 -s 1 2 3
```

Would run the multi-task setup (a training set of randomly shuffled MNIST and F-MNIST datasets) for 20 epochs, 3 times,
using seeds 1, 2, and 3.

Logging using wandb can be found under the DARK group at https://wandb.ai/ucl-dark/mnist-baseline-tests.