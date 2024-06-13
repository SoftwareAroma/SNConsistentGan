import os
import torch
import lpips
import numpy as np
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm
from scipy.stats import entropy

class BaseMetrics:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True

        # Initialize Inception model for FID and IS
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model = self.inception_model.to(self.device)

        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print('FID calculation produces singular product; adding %s to diagonal of cov estimates' % eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


    def calculate_activation_statistics(self, images, model, batch_size=50, dims=2048):
        """Calculation of the statistics used in the FID"""
        model.eval()
        act = np.empty((len(images), dims))
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(self.device)
                pred = model(batch)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                act[i:i + batch_size] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma


    def compute_is(self, images, splits=10):
        """Compute the Inception Score for a set of images"""
        self.inception_model.eval()
        resize = transforms.Resize((299, 299))
        images = torch.stack([resize(img) for img in images])
        with torch.no_grad():
            preds = self.inception_model(images.to(self.device)).softmax(dim=1).cpu().numpy()
        split_scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores), np.std(split_scores)


    def compute_lpips(self, real_images, fake_images):
        """Compute the LPIPS score between real and fake images"""
        self.lpips_model.eval()
        lpips_scores = []
        for real, fake in zip(real_images, fake_images):
            real = real.unsqueeze(0).to(self.device)
            fake = fake.unsqueeze(0).to(self.device)
            lpips_scores.append(self.lpips_model(real, fake).item())
        return np.mean(lpips_scores)


    def compute_fid(self, real_images, fake_images):
        """Compute the Frechet Inception Distance between real and fake images"""
        self.inception_model.eval()
        resize = transforms.Resize((299, 299))
        real_images = torch.stack([resize(img) for img in real_images])
        fake_images = torch.stack([resize(img) for img in fake_images])
        mu1, sigma1 = self.calculate_activation_statistics(real_images, self.inception_model)
        mu2, sigma2 = self.calculate_activation_statistics(fake_images, self.inception_model)
        fid_value = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value
    
    def save_metrics(self, metrics):
        """Save the metrics to a json file"""
        file = os.path.join(self.save_dir, 'metrics.json')
        torch.save(metrics, file)

