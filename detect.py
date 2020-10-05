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
    img = (img.view(1, img.shape[0], img.shape[1], img.shape[2]))  # Added batch size = 1
    return img


def get_vgg_model(model_path):
    vgg_model = torchvision.models.vgg16(pretrained=False)
    vgg_model.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=2, bias=True)
    vgg_model.load_state_dict(torch.load(model_path))
    return vgg_model


def has_tomatoes(img_path, model_path):
    model = get_vgg_model(model_path)
    print('Successfully loaded the model')
    img = get_img(img_path)
    print('Successfully loaded the image')

    y_pre = model(img)
    _, predicted = torch.max(y_pre.data, 1)
    classe = predicted.squeeze().numpy()
    if classe == 1:
        return True
    else:
        return False


if __name__ == '__main__':
    args = get_args()
    if has_tomatoes(args.img, args.model):
        print("There is some tomatoes !! Be careful ")
    else:
        print("No tomatoes detected here")
