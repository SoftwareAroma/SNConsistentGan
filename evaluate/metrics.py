"""
    Code From https://github.com/teboli/gan_metrics
"""
import os
from PIL import Image
import torch
from torch import nn
from torchvision import models
import numpy as np
import math
from . import losses
# from . import networks
from cleanfid import fid
import lpips
import torchvision.transforms as transforms

class InceptionScore(nn.Module):
    def __init__(self, state_dict=None, batch_size=32, splits=10):
        super(InceptionScore, self).__init__()
        self.batch_size = batch_size
        self.splits = splits
        self.name = 'inception'
        self.model = models.inception_v3(pretrained=True)  # model pretrained on ImageNet
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def forward(self, input):
        # Inception works only with 3x299x299 images. You can crop (like center crop or 10 crops)
        # the images or rescale them before feeding them into Inception.
        assert(input.shape[1] == 3 and input.shape[2] == 299 and input.shape[3] == 299)
        self.model.eval()

        n = input.shape[0]
        n_batches = int(math.ceil(float(n) / float(self.batch_size)))

        probs = []
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.batch_size
                end = min((i+1) * self.batch_size, n)
                input_split = input[start:end]
                probs.append(self.model(input_split).softmax(dim=1))
                
        probs = torch.cat(probs, dim=0)

        scores = []
        for i in range(self.splits):
            start = i * n // self.splits
            end = (i+1) * n // self.splits
            probs_split = probs[start:end]
            p = probs_split.mean(dim=0,keepdim=True).log()
            # kl = F.kl_div(probs_split, p, reduction='none')
            kl = probs_split*(probs_split.log() - p)
            kl = kl.sum(dim=1).mean()
            scores.append(kl.exp().item())

        return np.mean(scores), np.std(scores)


class FCNScore(nn.Module):
    def __init__(self, model=None, batch_size=10, num_classes=21):
        super(FCNScore, self).__init__()
        self.model = model if model is not None else models.segmentation.fcn_resnet50(pretrained=True)
        self.name = 'fcn'
        self.batch_size = batch_size
        self.num_classes = num_classes

    def load_and_preprocess_images(self, folder, target=False):
        images = []
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            if target:
                img = img.convert('L')  # Convert target images to grayscale
            else:
                img = img.convert('RGB')  # Convert input images to RGB
            img = transform(img)
            images.append(img)
        
        return torch.stack(images)

    def forward(self, input_folder, target_folder):
        # Load and preprocess images
        input_images = self.load_and_preprocess_images(input_folder)
        target_images = self.load_and_preprocess_images(target_folder, target=True)

        # Compute segmentation labels
        self.model.eval()
        n = input_images.shape[0]
        n_batches = int(math.ceil(float(n) / float(self.batch_size)))

        labels = []
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, n)
                input_split = input_images[start:end]
                outputs = self.model(input_split)['out']
                labels.append(outputs.argmax(dim=1))
        labels = torch.cat(labels, dim=0)

        return losses.label_score(labels, target_images, self.num_classes)
    

# class FIDScore(nn.Module):
#     """
#         Frechet Inception Distance
#     """
#     def __init__(self, state_dict=None, batch_size=32, device='cpu'):
#         super(FIDScore, self).__init__()
#         self.batch_size = batch_size
#         self.name = 'fid'
#         self.device = device
#         self.model = models.inception_v3(pretrained=True)
        
#         if state_dict is not None:
#             self.model.load_state_dict(state_dict)
            
#     def forward(self, folderA, folderB):
#         self.model.eval()
#         return fid.compute_fid(folderA, folderB, device=self.device, num_workers=0)

class FIDScore(nn.Module):
    """
    Frechet Inception Distance
    """
    def __init__(self, state_dict=None, batch_size=32, device='cpu'):
        super(FIDScore, self).__init__()
        self.batch_size = batch_size
        self.name = 'fid'
        self.device = device
        self.model = models.inception_v3(pretrained=True)
        
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
    
    def preprocess_images(self, folder):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        images = []
        for image in os.listdir(folder):
            image_path = os.path.join(folder, image)
            img = Image.open(image_path).convert('RGB')
            img = transform(img)
            images.append(img)
        return torch.stack(images)

    def forward(self, folderA, folderB):
        self.model.eval()

        # Preprocess images
        # imagesA = self.preprocess_images(folderA)
        # imagesB = self.preprocess_images(folderB)

        # Compute FID
        try:
            fid_score = fid.compute_fid(folderA, folderB, device=self.device, num_workers=0)
        except ValueError as e:
            print(f"Error computing FID: {e}")
            fid_score = None

        return fid_score
        
        
class KIDScore(nn.Module):
    """
        Kernel Inception Distance
    """
    def __init__(self, state_dict=None, batch_size=32, device='cpu'):
        super(KIDScore, self).__init__()
        self.batch_size = batch_size
        self.name = 'kid'
        self.device = torch.device(device)
        self.model = models.inception_v3(pretrained=True)
        
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            
    def forward(self, folderA, folderB):
        self.model.eval()
        return fid.compute_kid(folderA, folderB, device=self.device, num_workers=0)
    

class LPIPSScore(nn.Module):
    """
        Learned Perceptual Image Patch Similarity
    """
    def __init__(self, batch_size=32, net='alex'):
        super(LPIPSScore, self).__init__()
        self.batch_size = batch_size
        self.name = 'lpips'
        self.loss_fn = lpips.LPIPS(net=net)
        
    def load_and_preprocess_images(self, folder):
        images = []
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256 for LPIPS
            transforms.ToTensor(),
        ])
        
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
        
        return torch.stack(images)

    def forward(self, folderA, folderB):
        # Load and preprocess images
        imagesA = self.load_and_preprocess_images(folderA)
        imagesB = self.load_and_preprocess_images(folderB)

        # Compute LPIPS
        with torch.no_grad():
            lpips_scores = []
            for i in range(0, len(imagesA), self.batch_size):
                batchA = imagesA[i:i+self.batch_size]
                batchB = imagesB[i:i+self.batch_size]
                lpips_score = self.loss_fn(batchA, batchB)
                lpips_scores.append(lpips_score)

            lpips_scores = torch.cat(lpips_scores)
        
        return lpips_scores.mean().item()

        