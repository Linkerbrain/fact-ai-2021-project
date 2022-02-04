
export CUDA_VISIBLE_DEVICES=0
data=cifar100
epoch=200
arch='ResNet20-4'
python benchmark/cifar100_train.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=normal
#python benchmark/cifar100_train.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=crop
#python benchmark/cifar100_train.py --data=$data --arch=$arch --epochs=$epoch --aug_list='' --mode=aug

for aug_list in '3-1-7+43-18-18' '3-1-7' '43-18-18';
do
{
echo $aug_list
python -u benchmark/cifar100_train.py --data=$data --arch=$arch --epochs=$epoch --aug_list=$aug_list --mode=aug
}
done