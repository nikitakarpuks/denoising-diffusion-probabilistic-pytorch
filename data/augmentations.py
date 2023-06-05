from torchvision.transforms import Compose, ToTensor, Lambda


class TransformAugmentations:
    def __init__(self):
        self.transform = Compose([
            ToTensor(),  # turn into torch Tensor of shape CHW, divide by 255
            Lambda(lambda t: (t * 2) - 1),  # preprocessing
        ])

    def __call__(self, x):
        y = self.transform(x)
        return y
