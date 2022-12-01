import torch
import torch.nn as nn


class GoogLeNetLikeNet(nn.Module):
    def __init__(self, is_feature_extractor=False, use_batch_norm=False):
        super().__init__()
        """GoogLeNet inspired network architecture (DarkNet) to be pretrained on ImageNet 
            And to be used as backbone for the YOLO detector network `DetectorNet`.
            The convolutional layers of this network are the first 20 layers of the YOLOv1 Detector Network.
        """

        self.is_feature_extractor = is_feature_extractor

        self.features = self.create_conv_layers(use_batch_norm=use_batch_norm)
        if not is_feature_extractor:
            self.avg_pool = nn.AvgPool2d(7)
            self.classifier = self.create_classifier()


    def forward(self, x):
        x = self.features(x)

        if not self.is_feature_extractor:
            x = self.avg_pool(x)
            
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)

        return x

    def create_classifier(self):

        layers = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(1024, 1000)
        )
        
        return layers

    def create_conv_layers(self, use_batch_norm):
        layers = []

        # ----------1st block----------
        layers.append(nn.Conv2d(in_channels=3, out_channels=64, stride=2, kernel_size=7, padding=3))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(2,2))


        # ----------2nd block----------
        layers.append(nn.Conv2d(in_channels=64, out_channels=192, stride=1, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(192))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(2,2))


        # ----------3rd block----------
        layers.append(nn.Conv2d(in_channels=192, out_channels=128, stride=1, kernel_size=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(in_channels=256, out_channels=512, stride=1, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(2,2))


        # ----------4th block----------

        for _ in range(4):

            layers.append(nn.Conv2d(in_channels=512, out_channels=256, stride=1, kernel_size=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(256))
            layers.append(nn.LeakyReLU(0.1))

            layers.append(nn.Conv2d(in_channels=256, out_channels=512, stride=1, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(512))
            layers.append(nn.LeakyReLU(0.1))


        layers.append(nn.Conv2d(in_channels=512, out_channels=512, stride=1, kernel_size=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(in_channels=512, out_channels=1024, stride=1, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.MaxPool2d(2,2))


        # ----------5th block----------

        for _ in range(2):

            layers.append(nn.Conv2d(in_channels=1024, out_channels=512, stride=1, kernel_size=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(512))
            layers.append(nn.LeakyReLU(0.1))

            layers.append(nn.Conv2d(in_channels=512, out_channels=1024, stride=1, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(1024))
            layers.append(nn.LeakyReLU(0.1))

        layers = nn.Sequential(*layers)

        return layers


class DetectorNet(nn.Module):
    def __init__(self, feature_extractor, use_batch_norm=False, S=7, B=2, C=20):
        super().__init__()

        self.prediction_output_size = S * S * (B*5 + C)
        
        self.feature_extractor = feature_extractor

        self.features = self.create_conv_layers(use_batch_norm=use_batch_norm)
        self.conn = self.create_conn_layers()


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.features(x)

        x = x.view(x.shape[0], -1)
        
        x = self.conn(x)
        
        return x


    def create_conv_layers(self, use_batch_norm):
        layers = []

        layers.append(nn.Conv2d(in_channels=1024, out_channels=1024, stride=1, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(in_channels=1024, out_channels=1024, stride=2, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1))


        # ----------6th block----------

        layers.append(nn.Conv2d(in_channels=1024, out_channels=1024, stride=1, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(in_channels=1024, out_channels=1024, stride=1, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.1))


        layers = nn.Sequential(*layers)

        return layers


    def create_conn_layers(self):
        layers = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(.5),
            nn.Linear(4096, self.prediction_output_size),
            nn.Sigmoid()
        )

        return layers


class TinyNet(nn.Module):
    def __init__(self, imagenet=False, use_batch_norm=False, S=7, B=2, C=20):
        super().__init__()
        """Tiny version of the original YOLO detector network with only 9 
            convolutional layers (as oppsed to the 24 used in the bigger model)
        """

        self.features = self.create_conv_layers(use_batch_norm=use_batch_norm)

        if imagenet:
            self.classifier = self.create_imagenet_classifier()

        else:
            self.prediction_output_size = S * S * (B*5 + C)
            self.classifier = self.create_classifier()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x

    
    def create_classifier(self):
        layers = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(256*7*7, self.prediction_output_size),
            nn.Sigmoid()
        )
        
        return layers

    
    def create_imagenet_classifier(self):
        layers = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(256*3*3, 1000)
        )
        
        return layers


    def create_conv_layers(self, use_batch_norm):
        layers = []

        def create_block(in_channels, out_channels, downsample=True):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(16))
            layers.append(nn.LeakyReLU(0.1))
            if downsample:
                layers.append(nn.MaxPool2d(2,2))


        create_block(3, 16)
        [create_block(16 * 2**(i-1), 16 * 2**i) for i in range(1, 6)]
        create_block(16 * 2**5, 2 * 16 * 2**5, downsample=False)
        create_block(2 * 16 * 2**5, 16 * 2**4, downsample=False)


        layers = nn.Sequential(*layers)

        return layers


if __name__ == '__main__':
    import unittest
    
    detectornet_input = torch.zeros((1, 3, 224*2, 224*2))
    darknet_input = torch.zeros((1, 3, 224, 224))

    detectornet = DetectorNet(feature_extractor=GoogLeNetLikeNet(is_feature_extractor=True))
    darknet = GoogLeNetLikeNet(is_feature_extractor=False)

    tinynet_imagenet = TinyNet(imagenet=True)
    tinynet_yolo = TinyNet()

    class DetectorNetTest(unittest.TestCase):
        def test_output(self):
            pred = detectornet(detectornet_input)

            self.assertEqual(pred.shape, (1, 7*7 * (2*5 + 20)))


    class DarkNetTest(unittest.TestCase):
        def test_output(self):
            pred = darknet(darknet_input)
            self.assertEqual(pred.shape, (1, 1000))


    class TinyNetTest(unittest.TestCase):
        def test_imagenet_output(self):
            pred = tinynet_imagenet(darknet_input)
            self.assertEqual(pred.shape, (1, 1000))

        def test_detector_output(self):
            pred = tinynet_yolo(detectornet_input)
            self.assertEqual(pred.shape, (1, 7*7 * (2*5 + 20)))


    unittest.main()