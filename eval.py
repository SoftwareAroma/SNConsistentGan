import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from data import CreateDataLoader
from evaluate import metrics
from models import create_model

from options.eval_options import EvalOptions

preprocessing = {
    "inception": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },
}

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    return images

def preprocess_images(images, mean, std, size=(299, 299)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    images_tensor = torch.stack([transform(img) for img in images])
    return images_tensor

if __name__ == "__main__":
    opt = EvalOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    if opt.metric=='inception':
        metrics = [metrics.InceptionScore()]
    elif opt.metric=='fcn':
        metrics = [metrics.FCNScore()]
    elif opt.metric=='fid':
        metrics = [metrics.FIDScore()]
    elif opt.metric=='kid':
        metrics = [metrics.KIDScore()]
    elif opt.metric=='lpips':
        metrics = [metrics.LPIPSScore()]
    elif opt.metric=='all':
        metrics = [metrics.InceptionScore(), metrics.FIDScore(), metrics.LPIPSScore()]
        
    path_a = opt.dataroot + "/" + opt.testA
    path_b = opt.dataroot + "/" + opt.testB
    # Load images from test directories
    testA_images = load_images_from_folder(path_a)
    testB_images = load_images_from_folder(path_b)

    # Preprocess images
    mean, std = preprocessing['inception']['mean'], preprocessing['inception']['std']
    testA_tensors = preprocess_images(testA_images, mean, std)
    testB_tensors = preprocess_images(testB_images, mean, std)
    
    # Compute metrics
    if isinstance(metrics, list):
        results = {}
        for m in metrics:
            if m.name == 'inception':
                mean_score, std_score = m(testA_tensors)
                results['Inception'] = (mean_score, std_score)
            elif m.name == 'fid':
                fid_score = m(path_a, path_b)
                results['FID'] = fid_score
            elif m.name == 'lpips':
                lpips_score = m(path_a, path_b)
                results['LPIPS'] = lpips_score
            elif m.name == 'kid':
                kid_score = m(path_a, path_b)
                results['KID'] = kid_score
            elif m.name == 'fcn':
                fcn_score = m(path_a, path_b)
                results['FCN'] = fcn_score
        print(results)

