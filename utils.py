import torch
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset_NR, ImageDataset_FR, ImageDataset_csv, ImageDataset_group
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
    group_samples = batch[0]

    images = [sample['I'] for sample in group_samples]
    labels = [sample['mos'] for sample in group_samples]

    images_batch = torch.stack(images, dim=0)
    labels_batch = torch.tensor(labels)

    sample = {'I': images_batch, 'mos': labels_batch}

    return sample


def group_batch_collate_fn(batch):
    """
    Args:
        batch: batch_size-length List of group_size-length List of 2-length Dict.
               [[{'I': patches_1_1, 'mos': label_1_1}, {'I': patches_1_2, 'mos': label_1_2}, ...],
                [{'I': patches_2_1, 'mos': label_2_1}, {'I': patches_2_2, 'mos': label_2_2}, ...],
                ...,
                [{'I': patches_bs_1, 'mos': label_bs_1}, {'I': patches_bs_2, 'mos': label_bs_2}, ...]]
                patches_i_j.shape: Tensor(num_patch, c, h, w)
                labels_i_j.shape: int
    Returns:
        samples: 2-length Dict of Tensor.
                 {'I': images_batch, 'mos': labels_batch}
                  images_batch.shape: Tensor(batch_size * group_size, num_patch, c, h, w)
                  labels_batch.shape: Tensor(batch_size * group_size, label)
    """
    flat_batch = [sample for group in batch for sample in group]

    images = [sample['I'] for sample in flat_batch]
    labels = [torch.tensor(sample['mos']) for sample in flat_batch]

    images_batch = torch.stack(images, dim=0)
    labels_batch = torch.stack(labels, dim=0)

    samples = {'I': images_batch, 'mos': labels_batch}

    return samples


def set_dataset_NR(csv_file, bs, data_set, num_workers, preprocess, num_patch, test, set):

    data = ImageDataset_NR(
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


def set_dataset_FR(csv_file, bs, data_set, ref_set, num_workers, preprocess, num_patch, test, set):

    data = ImageDataset_FR(
        csv_file=csv_file,
        img_dir=data_set,
        ref_dir=ref_set,
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


def set_dataset_csv(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset_csv(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


def set_dataset_group(mos_dir, sr_dir, num_workers, preprocess, num_patch, test):
    dataset = ImageDataset_group(
        mos_dir=mos_dir,
        sr_dir=sr_dir,
        preprocess=preprocess,
        num_patch=num_patch,
        test=test)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=group_collate_fn,
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
        # Normalize((0.48145466, 0.4578275, 0.40821073),
        #           (0.26862954, 0.26130258, 0.27577711)),
    ])


def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        RandomHorizontalFlip(),
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073),
        #           (0.26862954, 0.26130258, 0.27577711)),
    ])


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
