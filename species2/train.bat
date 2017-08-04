echo 'This training process takes around 24 hours, please be patient and do not interrupt it...'

REM python train.py --train inception_v3
REM python train.py --fine_tune inception_v3
python train.py --train vgg19_bn
python train.py --fine_tune vgg19_bn
python train.py --train vgg16_bn
python train.py --fine_tune vgg16_bn
python train.py --train densenet169
python train.py --fine_tune densenet169
python train.py --train densenet161
python train.py --fine_tune densenet161


echo 'Training finished.'
