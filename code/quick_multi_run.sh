# Test impact of label levels
#python3 train.py --training_type multi-task_labels --label_level 1 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 2 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 3 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 4 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 5 --epochs 10 --seed 0 -v

# Compare kernel sizes for seed 0
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task --kernel_size 5 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task --kernel_size 7 --epochs 10 --seed 0 -v

# Compare kernel sizes for seed 1
#python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 1 -v
#python3 train.py --training_type multi-task --kernel_size 5 --epochs 10 --seed 1 -v
#python3 train.py --training_type multi-task --kernel_size 7 --epochs 10 --seed 1 -v

# Compare l2 norm regularisation lambdas
python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 -v
python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-1 -v
python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-2 -v
python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-3 -v
python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-4 -v
python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 --norm l2 --reg_lambda 1e-5 -v
