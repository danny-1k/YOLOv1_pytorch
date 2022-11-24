from torchvision.datasets import ImageNet

def get_imagetnet_dataset(train_transform=None, test_transform=None):
    if train_transform:
        train = ImageNet(root='data', split='train', transform=train_transform)
    else:
        train = ImageNet(root='data', split='train')

    if test_transform:
        test = ImageNet(root='data', split='val', transform=test_transform)
    else:
        test = ImageNet(root='data', split='val')


    return train, test