# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:49:58 2019

@author: hangting
"""
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import numpy as np
from Alexnet import Alexnet
from image_preprocessing import *
from torch.autograd import Variable

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x
    
    
class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output
    
    
def show_cam_on_image(img, mask):
#    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = (img/np.max(img)).transpose(1, 2, 0)
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    cv2.imwrite("heatmap.jpg", np.uint8(255 * heatmap))
    
    
class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		print(output.data)

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :] # size of CxHxW

		weights = np.mean(grads_val, axis = (2, 3))[0, :] # size of C
		cam = np.zeros(target.shape[1 : ], dtype = np.float32) # size of HxW

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam
    

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

#		self.model.features.zero_grad()
#		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output
    
    
if __name__ == '__main__':
    # load the model
    load_path = './weights/alexnet_weight.pt'
    model = Alexnet()
    model.load_state_dict(torch.load(load_path))
    
    grad_cam = GradCam(model = model, target_layer_names = ["12"], use_cuda=True)
    input, img = load_data('PD_data.npy')

	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
    target_index = 0

    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model = model, use_cuda=True)
    gb = gb_model(input, index=target_index)
    utils.save_image(torch.from_numpy(gb), 'gb.jpg')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
