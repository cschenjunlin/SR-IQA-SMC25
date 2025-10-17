import torch
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset_qonly, ImageDataset_pl
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def group_collate_fn(batch):
    """
    Convert batch data from 1-length list to tensor.
    """
    batch_samples = batch[0]

    images = [sample['I'] for sample in batch_samples]
    labels = [sample['mos'] for sample in batch_samples]

    images_batch = torch.stack(images, dim=0)
    labels_batch = torch.tensor(labels)

    return {'I': images_batch, 'mos': labels_batch}


def set_dataset_qonly(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, set):

    data = ImageDataset_qonly(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        set=set,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_dataset_pl(csv_file, data_set, num_workers, preprocess, num_patch, test, set, pseudo_label):

    data = ImageDataset_pl(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        pseudo_label=pseudo_label,
        set=set,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=1, shuffle=shuffle, collate_fn=group_collate_fn,
                        pin_memory=True, num_workers=num_workers)

    return loader


class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
