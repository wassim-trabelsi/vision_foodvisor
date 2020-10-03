import numpy as np
import torch
from torch.utils.data import DataLoader

# Calculate the accuracy to evaluate the model
def accuracy(dataset, model):
    model.eval()
    confusion = np.zeros((2,2), dtype = int)
    list_of_errors = []
    with torch.no_grad():
        correct = 0
        total = 0
        dataloader = DataLoader(dataset, batch_size= 1)
        for idx, (images, labels, imagenames) in enumerate(dataloader):
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            confusion[predicted.cpu().numpy()[0],labels.cpu().numpy()[0]]+=1
            correct += (predicted == labels).sum()
            if (predicted == labels).sum() ==0:
                list_of_errors.append((imagenames[0],predicted, labels))
    print('Accuracy of the model : {:.2f} %'.format(100*correct.item()/ len(dataset)))
    print(confusion)
    return confusion, list_of_errors