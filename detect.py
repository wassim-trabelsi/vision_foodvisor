import argparse
import torchvision


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
    predict = model(img)
    print(predict)


if __name__ == '__main__':
    args = get_args()
    vgg_model = torchvision.models.vgg16(pretrained=False)
    vgg_model.load_state_dict(torch.load(args.model))

    print('Successfully loaded state dict')
    img = get_img(args.img)

    print('Successfully loaded the image')
    has_tomatoes(vgg_model, img)
