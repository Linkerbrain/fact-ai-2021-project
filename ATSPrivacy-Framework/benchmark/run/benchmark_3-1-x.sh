
export CUDA_VISIBLE_DEVICES=0
data=cifar100
epoch=200
arch='ResNet20-4'
for aug_list in '3-1-39';
do
{
echo $aug_list
python -u benchmark/cifar100_train.py --data=$data --arch=$arch --epochs=$epoch --aug_list=$aug_list --mode=aug
python -u benchmark/cifar100_attack.py --data=$data --arch=$arch --epochs=$epoch --aug_list=$aug_list --mode=aug --optim='inversed'
}
done