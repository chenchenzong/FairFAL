## Run 

```
### FMNIST
python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset fmnist_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset fmnist_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 100 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

### CIFAR-10
python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 100 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

### CIFAR-100
python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset cifar100_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 100 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset cifar100_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 100 --num_users 10 --frac 1.0 --num_classes 100 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

### OctMNIST
python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset octmnist --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

### DermaMNIST
python main.py --gpu 0 --seed 1 --al_method fairfal --model cnn4conv --dataset dermamnist --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05

```

