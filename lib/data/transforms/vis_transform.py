import torchvision.transforms as T

from transforms import RandomErasing, RandomPatch, ColorSpaceConvert, ColorAugmentation, RandomBlur
from augmix import AugMix

if __name__ == '__main__':
    from PIL import Image
    import cv2
    img_path = '/home/zxy/data/ReID/vehicle/AIC20_ReID/image_query/000345.jpg'
    img = Image.open(img_path).convert('RGB')

    transform = T.Compose([
        T.Resize([256, 256]),
        T.RandomHorizontalFlip(0.0),
        T.Pad(0),
        T.RandomCrop([256, 256]),
        AugMix(prob=0.5),
        RandomBlur(p=1.0),
    ])
    canvas = transform(img)
    cv2.imwrite('test.jpg', canvas[:, :, ::-1])