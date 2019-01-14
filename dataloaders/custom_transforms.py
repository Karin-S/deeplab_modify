import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
# import cv2

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class Resize_normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        h, w, _ = img.shape
        ratio = 513. / np.max([w, h])
        resized = cv2.resize(img, (int(ratio * w), int(ratio * h)))
        resized = resized / 127.5 - 1.
        if w < h:
            pad_x = int(513 - resized.shape[1])
            resized2 = np.pad(resized, ((0, 0), (0, pad_x), (0, 0)), mode='constant')
        elif w >= h:
            pad_y = int(513 - resized.shape[0])
            resized2 = np.pad(resized, ((0, pad_y), (0, 0), (0, 0)), mode='constant')

        resized2 = resized2.transpose(2, 0, 1)
        resized3 = torch.from_numpy(resized2).float()

        return resized3


class Resize_normalize_train(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img1 = np.array(img).astype(np.float32)

        h, w, _ = img1.shape
        ratio = 513. / np.max([w, h])
        resized = img.resize((int(ratio * w), int(ratio * h)), Image.BILINEAR)
        resized = np.array(resized).astype(np.float32)
        resized = resized / 127.5 - 1.
        if w < h:
            pad_x = int(513 - resized.shape[1])
            resized2 = np.pad(resized, ((0, 0), (0, pad_x), (0, 0)), mode='constant')
        elif w >= h:
            pad_y = int(513 - resized.shape[0])
            resized2 = np.pad(resized, ((0, pad_y), (0, 0), (0, 0)), mode='constant')
        resized2 = resized2.transpose(2, 0, 1)
        resized3 = torch.from_numpy(resized2).float()

        resized_m = mask.resize((int(ratio * w), int(ratio * h)), Image.NEAREST)
        resized_m = np.array(resized_m).astype(np.float32)
        if w < h:
            pad_x = int(513 - resized_m.shape[1])
            resized2_m = np.pad(resized_m, ((0, 0), (0, pad_x)), mode='constant')
        elif w >= h:
            pad_y = int(513 - resized_m.shape[0])
            resized2_m = np.pad(resized_m, ((0, pad_y), (0, 0)), mode='constant')
        resized3_m = torch.from_numpy(resized2_m).float()

        return {'image': resized3,
                'label': resized3_m}


# class Resize_normalize_val(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         mean (tuple): means for each channel.
#         std (tuple): standard deviations for each channel.
#     """
#     def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         id = sample['id']
#
#         # img1 = np.array(img).astype(np.float32)
#
#         h = img.height
#         w = img.width
#         ratio = 513. / np.max([w, h])
#         size = (int(ratio * w), int(ratio * h))
#         resized = img.resize( size, Image.BILINEAR )
#         resized = np.array(resized).astype(np.float32)
#         resized = resized / 127.5 - 1.
#         if w < h:
#             pad_x = int(513 - resized.shape[1])
#             resized2 = np.pad(resized, ((0, 0), (0, pad_x), (0, 0)), mode='constant')
#         elif w >= h:
#             pad_y = int(513 - resized.shape[0])
#             resized2 = np.pad(resized, ((0, pad_y), (0, 0), (0, 0)), mode='constant')
#         resized2 = resized2.transpose(2, 0, 1)
#         resized3 = torch.from_numpy(resized2).float()
#
#         resized_m = mask.resize(size, Image.NEAREST)
#         resized_m = np.array(resized_m).astype(np.float32)
#         if w < h:
#             pad_x = int(513 - resized_m.shape[1])
#             resized2_m = np.pad(resized_m, ((0, 0), (0, pad_x)), mode='constant')
#         elif w >= h:
#             pad_y = int(513 - resized_m.shape[0])
#             resized2_m = np.pad(resized_m, ((0, pad_y), (0, 0)), mode='constant')
#         resized3_m = torch.from_numpy(resized2_m).float()
#
#         return {'image': resized3,
#                 'label': resized3_m,
#                 'id': id}


class Resize_normalize_val(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        id = sample['id']
        h = img.height
        w = img.width
        ratio = 513. / np.max([w, h])
        size = (int(ratio * w), int(ratio * h))
        resized = img.resize(size, Image.BILINEAR)      #Image object
        resized = np.array(resized).astype(np.float32)
        resized = resized / 127.5 - 1.
        if w < h:
            pad_x = int(513 - resized.shape[1])
            resized2 = np.pad(resized, ((0, 0), (0, pad_x), (0, 0)), mode='constant')
        elif w >= h:
            pad_y = int(513 - resized.shape[0])
            resized2 = np.pad(resized, ((0, pad_y), (0, 0), (0, 0)), mode='constant')
        resized2 = resized2.transpose(2, 0, 1)
        resized3 = torch.from_numpy(resized2).float()
        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        return {'image': resized3,
                'label':  mask,
                'id': id}


class compare(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        id = sample['id']

        img = np.array(img).astype(np.float32)
        img = torch.from_numpy(img).float()
        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label':  mask,
                'id': id}

# totally same as Resize_normalize_train with cv2
class Resize_normalize_train2(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        h, w, _ = img.shape
        ratio = 513. / np.max([w, h])
        resized = cv2.resize(img, (int(ratio * w), int(ratio * h)))
        resized = resized / 127.5 - 1.
        if w < h:
            pad_x = int(513 - resized.shape[1])
            resized2 = np.pad(resized, ((0, 0), (0, pad_x), (0, 0)), mode='constant')
        elif w >= h:
            pad_y = int(513 - resized.shape[0])
            resized2 = np.pad(resized, ((0, pad_y), (0, 0), (0, 0)), mode='constant')
        resized2 = resized2.transpose(2, 0, 1)
        resized3 = torch.from_numpy(resized2).float()

        resized_m = cv2.resize(mask, (int(ratio * w), int(ratio * h)))
        if w < h:
            pad_x = int(513 - resized_m.shape[1])
            resized2_m = np.pad(resized_m, ((0, 0), (0, pad_x)), mode='constant')
        elif w >= h:
            pad_y = int(513 - resized_m.shape[0])
            resized2_m = np.pad(resized_m, ((0, pad_y), (0, 0)), mode='constant')
        resized3_m = torch.from_numpy(resized2_m).float()
        return {'image': resized3,
                'label': resized3_m,
                'id': sample['id']}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}