# Test impact of label levels
#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 1 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 2 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 3 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 4 --epochs 10 --seed 0 -v

# Compare kernel sizes for seed 0
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task --kernel_size 5 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task --kernel_size 7 --epochs 10 --seed 0 -v

# Compare kernel sizes for seed 1
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 1 -v
#python3 train.py --training_type multi-task --kernel_size 5 --epochs 10 --seed 1 -v
#python3 train.py --training_type multi-task --kernel_size 7 --epochs 10 --seed 1 -v

# Compare l2 norm regularisation lambdas
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-1 -v
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-2 -v
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-3 -v
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-4 -v
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-5 -v

#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v --norm l2 --reg_lambda 0
#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v --norm l2 --reg_lambda 1e-7
#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v --norm l2 --reg_lambda 1e-6
#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v --norm l2 --reg_lambda 1e-5

## Baselines training on one dataset
#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v -o MNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 1 -v -o MNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 2 -v -o MNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 3 -v -o MNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 4 -v -o MNIST
#
#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v -o FMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 1 -v -o FMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 2 -v -o FMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 3 -v -o FMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 4 -v -o FMNIST
#
#python3 train.py --training_type multi-task --epochs 10 --seed 0 -v -o KMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 1 -v -o KMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 2 -v -o KMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 3 -v -o KMNIST
#python3 train.py --training_type multi-task --epochs 10 --seed 4 -v -o KMNIST


# Baselines training on two datasets
python3 train.py --training_type multi-task --epochs 10 --seed 0 -v -o MNIST FMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 1 -v -o MNIST FMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 2 -v -o MNIST FMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 3 -v -o MNIST FMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 4 -v -o MNIST FMNIST

python3 train.py --training_type multi-task --epochs 10 --seed 0 -v -o MNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 1 -v -o MNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 2 -v -o MNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 3 -v -o MNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 4 -v -o MNIST KMNIST

python3 train.py --training_type multi-task --epochs 10 --seed 0 -v -o FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 1 -v -o FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 2 -v -o FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 3 -v -o FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 4 -v -o FMNIST KMNIST


# Baselines on all three
python3 train.py --training_type multi-task --epochs 10 --seed 0 -v -o MNIST FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 1 -v -o MNIST FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 2 -v -o MNIST FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 3 -v -o MNIST FMNIST KMNIST
python3 train.py --training_type multi-task --epochs 10 --seed 4 -v -o MNIST FMNIST KMNIST