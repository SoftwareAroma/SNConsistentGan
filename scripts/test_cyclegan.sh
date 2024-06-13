set -ex
python test.py --dataroot ./datasets/cycle_gan --name horses_cyclegan --model cycle_gan --phase test --no_dropout
