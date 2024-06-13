set -ex
python train.py --dataroot ./datasets/cycle_gan --name horeses_cyclegan --model cycle_gan --pool_size 50 --no_dropout

# change dataset to cityscapes and model name to cityscapes_cyclegan