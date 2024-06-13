
#### Connection Error:HTTPConnectionPool
Similar error messages include “Failed to establish a new connection/Connection refused”.

Please start the visdom server before starting the training:
```bash
python -m visdom.server
```
To install the visdom, you can use the following command:
```bash
pip install visdom
```
You can also disable the visdom by setting `--display_id 0`.

#### My PyTorch errors on CUDA related code.
Try to run the following code snippet to make sure that CUDA is working (assuming using PyTorch >= 0.4):
```python
import torch
torch.cuda.init()
print(torch.randn(1, device='cuda'))
```

If you met an error, it is likely that your PyTorch build does not work with CUDA, e.g., it is installl from the official MacOS binary, or you have a GPU that is too old and not supported anymore. You may run the the code with CPU using `--device_ids -1`.


#### ValueError: empty range for randrange()
Similar error messages include "ConnectionRefusedError: [Errno 111] Connection refused"

It is related to data augmentation step. It often happens when you use `--resize_or_crop crop`. The program will crop random `fineSize x fineSize` patches out of the input training images. But if some of your image sizes (e.g., `256x384`) are smaller than the `fineSize` (e.g., 512), you will get this error. A simple fix will be to use other data augmentation methods such as `--resize_and_crop` or `scale_width_and_crop`.  Our program will automatically resize the images according to `loadSize` before apply `fineSize x fineSize` cropping. Make sure that `loadSize >= fineSize`.


#### Can I continue/resume my training
You can use the option `--continue_train`. Also set `--epoch_count` to specify a different starting epoch count.

#### Why does my training loss not converge
Many GAN losses do not converge (exception: WGAN, WGAN-GP, etc. ) due to the nature of minimax optimization. For DCGAN and LSGAN objective, it is quite normal for the G and D losses to go up and down. It should be fine as long as they do not blow up.

#### How can I make it work for my own data (e.g., 16-bit png, tiff, hyperspectral images)
The current code only supports RGB and grayscale images. If you would like to train the model on other data types, please follow the following steps:

- change the parameters `--input_nc` and `--output_nc` to the number of channels in your input/output images.
- Write your own custom data loader (It is easy as long as you know how to load your data with python). If you write a new data loader class, you need to change the flag `--dataset_mode` accordingly. Alternatively, you can modify the existing data loader.

- If you use visdom and HTML to visualize the results, you may also need to change the visualization code.

#### Multi-GPU Training
You can use Multi-GPU training by setting `--gpu_ids` (e.g., `--gpu_ids 0,1,2,3` for the first four GPUs on your machine.) To fully utilize all the GPUs, you need to increase your batch size. Try `--batch_size 4`, `--batch_size 16`, or even a larger batch_size. Each GPU will process batch_size/#GPUs images. The optimal batch size depends on the number of GPUs you have, GPU memory per GPU, and the resolution of your training images.

We also recommend that you use the instance normalization for multi-GPU training by setting `--norm instance`. The current batch normalization might not work for multi-GPUs as the batchnorm parameters are not shared across different GPUs. Advanced users can try [synchronized batchnorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).


#### Can I run the model on CPU
Yes, you can set `--gpu_ids -1`. See [training/test tips](TIPS.md) for more details.


#### Out of memory
CycleGAN is more memory-intensive than pix2pix as it requires two generators and two discriminators. If you would like to produce high-resolution images, you can do the following.

- During training, train CycleGAN on cropped images of the training set. Please be careful not to change the aspect ratio or the scale of the original image, as this can lead to the training/test gap. You can usually do this by using `--resize_or_crop crop` option, or `--resize_or_crop scale_width_and_crop`.

- Then at test time, you can load only one generator to produce the results in a single direction. This greatly saves GPU memory as you are not loading the discriminators and the other generator in the opposite direction. You can probably take the whole image as input. You can do this using `--model test --dataroot [path to the directory that contains your test images (e.g., ./datasets/horse2zebra/trainA)] --model_suffix _A --resize_or_crop none`. You can use either `--resize_or_crop none` or `--resize_or_crop scale_width --fineSize [your_desired_image_width]` to produce high-resolution images.


#### The color gets inverted from the beginning of training
The authors also observe that the generator unnecessarily inverts the color of the input image early in training, and then never learns to undo the inversion. In this case, you can try two things.

- First, try using identity loss `--lambda_identity 1.0` or `--lambda_identity 0.1`. the identity loss makes the generator to be more conservative and make fewer unnecessary changes. However, because of this, the change may not be as dramatic.

- Second, try smaller variance when initializing weights by changing `--init_gain`. smaller variance in weight initialization results in less color inversion.

