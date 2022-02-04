
export CUDA_VISIBLE_DEVICES=0
data=cifar100
epoch=200
arch='ResNet20-4'
python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed'
#python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed-sim-out'
#python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed-adam-L1'
#python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed-adam-L2'
#python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed-sgd-sim'
#python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed-simlocal'
#python benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal --optim='inversed-LBFGS-sim'

for aug_list in '3-1-7+43-18-18' '3-1-7' '43-18-18';
do
{
echo $aug_list
python -u benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list=$aug_list --mode=aug --optim='inversed'
}
done