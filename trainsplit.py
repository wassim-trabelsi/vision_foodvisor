from foodLoader import FoodLoader
from torchvision import transforms

tr_train = transforms.Compose(
    [transforms.RandomChoice([transforms.Resize(256), transforms.Resize(224), transforms.Resize(288)]),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

tr_val = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = FoodLoader(transform=tr_train, mode='train')
val_dataset = FoodLoader(transform=tr_val, mode='val')
test_dataset = FoodLoader(transform=tr_val, mode='test')
