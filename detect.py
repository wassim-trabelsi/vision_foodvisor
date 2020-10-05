import argparse
import torchvision
import torch
import torch.nn as nn
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description='Tomato detector ')
    parser.add_argument('--img', '-i',
                        help="image path", required=True)
    parser.add_argument('--model', '-m',
                        default='output/vgg_model1.pth',
                        help='model path')

    return parser.parse_args()


tr_val = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                         torchvision.transforms.CenterCrop(224),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])


def get_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = tr_val(img)
    return img


def has_tomatoes(model, img):
    y_pre = model(img)
    _, predicted = torch.max(y_pre.data, 1)
    classe = predicted.squeeze().numpy()
    if classe == 1:
        return True
    else:
        return False


if __name__ == '__main__':
    args = get_args()
    vgg_model = torchvision.models.vgg16(pretrained=False)
    vgg_model.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=2, bias=True)
    vgg_model.load_state_dict(torch.load(args.model))
    print('Successfully loaded state dict')
    img = get_img(args.img)
    img = (img.view(1, img.shape[0], img.shape[1], img.shape[2]))
    print('Successfully loaded the image')

    if has_tomatoes(vgg_model, img):
        print("There is some tomatoes !! Be careful ")
    else:
        print("No tomatoes detected here")