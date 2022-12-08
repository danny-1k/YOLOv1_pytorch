from torchvision.datasets import ImageNet
from torch.utils.data import Dataset
from torchvision.datasets.voc import VOCDetection
from transforms import detection_transforms
from utils import label_from_voc

def get_imagetnet_dataset(train_transform=None, test_transform=None):
    if train_transform:
        train = ImageNet(root='../data/imagenet', split='train', transform=train_transform)
    else:
        train = ImageNet(root='../data/imagenet', split='train')

    if test_transform:
        test = ImageNet(root='../data/imagenet', split='val', transform=test_transform)
    else:
        test = ImageNet(root='../data/imagenet', split='val')


    return train, test


class PascalVocDataset(Dataset):
    def __init__(self, train=True, download=False, B=2, S=7, transform=None) -> None:
        super().__init__()

        self.B = B
        self.S = S
        # classes are 20 in VOC

        if train:
            if not transform:
                self.transform = detection_transforms['train']
            else:
                self.transform = transform
        else:
            if not transform:
                self.transform = detection_transforms['test']
            else:
                self.transform = transform

        self.dataset = VOCDetection(
            root='../data/pascalvoc', year='2007',
            image_set='train' if train else 'val',
            transform=self.transform,
            download=download,
        )


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, annotation = self.dataset[index]
        annotation = annotation['annotation']

        w = int(annotation['size']['width'])
        h = int(annotation['size']['height'])

        annotations = annotation['object']

        y = label_from_voc(annotations=annotations, width=w, height=h, B=self.B, S=self.S)

        return x, y


if __name__ == '__main__':
    ptrain = PascalVocDataset(train=True)
    ptest = PascalVocDataset(train=False)
    x, y = ptest[0]

    print(y.shape)