#!/bin/sh
echo 'This training process takes around 24 hours, please be patient and do not interrupt it...'

python train.py --train dense161 --freeze
python train.py --train dense161
python train.py --train dense121 --freeze
python train.py --train dense121
python train.py --train dense201 --freeze
python train.py --train dense201
python train.py --train dense169 --freeze
python train.py --train dense169
python train.py --train res50 --freeze
python train.py --train res50
python train.py --train res101 --freeze
python train.py --train res101
python train.py --train res152 --freeze
python train.py --train res152
python train.py --train inceptionv3 --freeze
python train.py --train inceptionv3
python train.py --train vgg19bn --freeze
python train.py --train vgg19bn
python train.py --train vgg16bn --freeze
python train.py --train vgg16bn

echo 'Training finished.'
