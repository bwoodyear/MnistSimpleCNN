#python3 train.py --training_type multi-task_labels --label_level 1 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 2 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 3 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 4 --epochs 10 --seed 0 -v
#python3 train.py --training_type multi-task_labels --label_level 5 --epochs 10 --seed 0 -v

python3 train.py --training_type multi-task --kernel_size 3 --epochs 10 --seed 0 -v
python3 train.py --training_type multi-task --kernel_size 5 --epochs 10 --seed 0 -v
python3 train.py --training_type multi-task --kernel_size 7 --epochs 10 --seed 0 -v