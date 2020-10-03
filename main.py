import torch
from train import train
from config import num_epochs, batch_size, criterion, optimizer
from trainsplit import train_dataset, val_dataset, test_dataset
from model import vgg_model
from evaluate import accuracy
from grad_cam import grad_cam, show_cam_on_image
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Tomato detector ')
    parser.add_argument('--train', '-t', action='store_true',
                        help="Train a new network")
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='Evaluate accuracy on train val and test')
    parser.add_argument('--cam', '-c', action='store_true',
                        help="Visualize GRAD CAM")
    return parser.parse_args()

EVALUATE = False
SHOW_GRAD_CAM = False

if __name__ == '__main__':
    args = get_args()

    TRAIN = args.train
    EVALUATE = args.evaluate
    SHOW_GRAD_CAM = args.cam
    if TRAIN:
        train_error, val_error, train_acc, val_acc = train(num_epochs, batch_size, criterion, optimizer, vgg_model,
                                                           train_dataset, val_dataset)
        torch.save(vgg_model.state_dict(), 'output2.pth')

        # plot the training error wrt. the number of epochs:
        plt.plot(range(1, num_epochs + 1), train_error, c='blue')
        plt.plot(range(1, num_epochs + 1), val_error, c='orange')
        plt.xlabel("num_epochs")
        plt.ylabel("error")
        plt.title("Visualisation of convergence")
        plt.savefig('Visualisation_of_convergence2.png')
    else:
        vgg_model.load_state_dict(torch.load('vgg_model1.pth'))
        print('Successfully loaded state dict')
        im = cv2.imread("Visualisation_of_convergence.png")
        plt.imshow(im)
        plt.show()

    if EVALUATE:
        plt.figure()
        print('TRAIN ACCURACY')
        confusion, list_of_errors = accuracy(train_dataset, vgg_model)
        heatmap = sns.heatmap(confusion, annot=True, cmap="Blues")
        heatmap.set_xlabel('ground truth')
        heatmap.set_ylabel('predicted')
        plt.title('Train confusion')

        plt.figure()
        print('VAL ACCURACY')
        confusion, list_of_errors = accuracy(val_dataset, vgg_model)
        heatmap = sns.heatmap(confusion, annot=True, cmap="Blues")
        heatmap.set_xlabel('ground truth')
        heatmap.set_ylabel('predicted')
        plt.title('Val confusion')

        plt.figure()
        print('TEST ACCURACY')
        confusion, list_of_errors = accuracy(test_dataset, vgg_model)
        heatmap = sns.heatmap(confusion, annot=True, cmap="Blues")
        heatmap.set_xlabel('ground truth')
        heatmap.set_ylabel('predicted')
        plt.title('test confusion')
        plt.show()


    if SHOW_GRAD_CAM:
        target_layer = '29'  # after the ReLU ('rectified convolutional feature map')
        desired_class = 1

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        for (X_batch, y_real, imagename) in train_loader:
            L_grad_cam = grad_cam(vgg_model, X_batch.to('cuda'), target_layer, desired_class)
            new_image = X_batch[0].numpy()
            new_image = np.moveaxis(new_image, 0, 2)
            show_cam_on_image(new_image, L_grad_cam)
            val_dataset.showanns(imagename[0])
            plt.show()
            break

