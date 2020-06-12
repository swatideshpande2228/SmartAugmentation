# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:00:14 2020

@author: Administrator
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import numpy as np
import pandas as pd
import cv2
import os
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import random
from torch.utils.data.sampler import SubsetRandomSampler
import glob
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
 
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Hyperparameters for our network
input_size = 784
hidden_sizes = [64, 128]
output_size = 10
# Build a feed-forward network
NetworkB = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(NetworkB)


def Optimizer_netA(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer_netA = optim.LBFGS([input_img.requires_grad_()])
    return optimizer_netA

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer_netA = Optimizer_netA(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer_netA.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss_netA = style_score + content_score
            loss_netA.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer_netA.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


#def save_augment_images(img):
 #   cv2.imwrite('C:\Swati\Thesis Project\Code\Swati_Code\Dataset\Augmented Images\Aug1.jpg', img)
  #  print('Augmented images saved successfully')

#save_augment_images(output)

def image_loader(image_name):
    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    Orinal_loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor(),                 #convert the image to a Tensor
        ])
    image = Image.open(image_name)
    image = Orinal_loader(image).unsqueeze(0)
    return image.to(device, torch.float)

dir_name="C:/Swati/Thesis Project/Code/Swati_Code/Dataset"
base_files=os.listdir(dir_name)
Files = []
for file in base_files:
    Images = os.path.join(dir_name, file)
    Files.append(Images)

epoch = 10

for i in range(epoch):
    total_loss = 0
    print("Augmentation starts")
    Original_img = image_loader(random.choice(Files))
    Augment_img = image_loader(random.choice(Files))
    
    input_img = Original_img.clone()
    #input_img = torch.randn(Original_img.data.size(), device=device)

    print(Original_img.shape)
    print(Augment_img.shape)

    assert Original_img.size() == Augment_img.size(), \
        "we need to import style and content images of the same size"

    unloader = transforms.ToPILImage()

    plt.ion()

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated

    plt.figure()
    imshow(Original_img, title='Random Image_1')

    plt.figure()
    imshow(Augment_img, title='Random Image_2')
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, Original_img,
                            Augment_img, input_img)

    plt.figure()
    imshow(output, title='Output Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()
    
    print("augment data shape", output.shape)
    outputs = NetworkB(output)

    optimizer_netB = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    criterion_netB = nn.CrossEntropyLoss()
        
    loss_netB = criterion_netB(outputs)
    loss_netB.backward()
    optimizer_netB.step()
    
    #loss = self.alpha * loss_netA + self.beta * loss_netB
    #total_loss += loss.data[0]

# zero the parameter gradients
    optimizer_netB.zero_grad()


# print statistics
    running_loss += loss_netB.item()

    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 2000))
    running_loss = 0.0

print('Finished Training')