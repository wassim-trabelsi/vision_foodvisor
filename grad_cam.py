import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def show_cam_on_image(image, feature_map):
    ### first construct the heatmap
    heatmap = cv2.applyColorMap(np.uint8(np.round(255 * feature_map)), cv2.COLORMAP_HSV)
    heatmap = np.float64(heatmap) / 255

    ### normalize the image
    image -= image.min()
    image /= (image.max() + 1e-6)

    ### final image
    final_image = heatmap + image
    final_image -= final_image.min()  # rescaling in [0, 1] (1)
    final_image /= (final_image.max() + 1e-6)  # rescaling in [0, 1] (2)

    ### plot
    plt.figure(figsize=(15, 15))
    plt.imshow(final_image)


class Hook():
    '''Class that stores the input and output of a layer during forward/backward pass.'''

    def __init__(self, module):
        self.hook = module.register_hook(self.save_gradient)

    def save_gradient(self, gradient):
        self.gradients = gradient


def grad_cam(model, image, target_layer, desired_class):
    ### set the model in evaluation mode
    model.eval()

    ### forward-propagate the image throw the 'features' module
    output_features = image  # input image
    for layer_index, module in model.features._modules.items():  # loop over the layers of model.features
        output_features = module(output_features)  # feed the network
        if layer_index == target_layer:  # if it is the layer of interest, we apply 'hook' function to store in/ouputs
            hook = Hook(output_features)  # hook the layer of interest
            output_target_layer = output_features[0, :, :, :].detach().cpu().numpy()  # output for the layer of interest

    ### forward-propagate the image throw the 'classifier' module
    output = model.classifier(output_features.view(output_features.size(0), -1)).cpu()  # feed the 'classifier' module

    ### set the gradients to zero
    model.features.zero_grad()  # sets gradients of all model parameters to zero (in 'features' module)
    model.classifier.zero_grad()  # sets gradients of all model parameters to zero (in 'classifier' module)

    ### one hot encoding for the desired class
    one_hot = torch.tensor(np.zeros(output.size()))  # initialize one hot encoding
    one_hot[0, desired_class] = 1  # set the desired class to 1
    one_hot = torch.sum(one_hot * output)  # element wise multiplication in order to achieve grad
    one_hot.backward()  # backpropagate the message

    ### compute the coarse Grad-CAM localization
    gradient_values = hook.gradients.cpu().numpy()[0, :, :, :]  # extract the gradient values using the hook class
    global_average_pooled_gradients = gradient_values.reshape((gradient_values.shape[0], -1)).mean(axis=1)
    L_grad_cam = np.zeros((14, 14))
    for i in range(512):
        L_grad_cam += global_average_pooled_gradients[i] * output_target_layer[i]
    L_grad_cam = np.maximum(L_grad_cam, 0)  # apply ReLU

    ### resize the L_grad_cam
    L_grad_cam = cv2.resize(L_grad_cam, (224, 224))
    L_grad_cam -= L_grad_cam.min()  # scale
    L_grad_cam /= (L_grad_cam.max() + 1e-6)

    return L_grad_cam
