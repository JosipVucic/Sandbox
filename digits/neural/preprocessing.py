import math

import torch
from PIL import Image
from torchvision.transforms import v2


def threshold_image(image, threshold=0.5):
    return (image > threshold).float()


def preprocess_image(file):
    """Preprocesses an image file, so it can be used with the GACNN model.
    :param file: image file to be opened with Pillow
    :return: input for the GACNN model, containing the processed image"""
    with Image.open(file) as original:
        transform = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Grayscale(),
                                v2.functional.invert,
                                v2.Resize((28, 28), antialias=None),
                                threshold_image])

        img = transform(original)

        # Identify rows and columns with all zeros
        non_zero_rows = torch.any(img, dim=2).squeeze()
        non_zero_cols = torch.any(img, dim=1).squeeze()

        # Keep only the rows/columns that contain the digit
        img = img[:, non_zero_rows, :][:, :, non_zero_cols]

        # Resize the image to fit into a 20x20 box while maintaining the aspect ratio
        aspect_ratio = img.size(2) / img.size(1)
        if img.size(2) > img.size(1):
            img = v2.functional.resize(img, [int(round(20 / aspect_ratio)), 20], antialias=None)
        else:
            img = v2.functional.resize(img, [20, int(round(20 * aspect_ratio))], antialias=None)

        # Calculate padding
        padding_h = [int(math.ceil((28 - img.size(2)) / 2.0)), int(math.floor((28 - img.size(2)) / 2.0))]
        padding_w = [int(math.ceil((28 - img.size(1)) / 2.0)), int(math.floor((28 - img.size(1)) / 2.0))]
        padding = padding_w + padding_h

        # Pad the image with ones to reach the final size of 28x28
        img = v2.functional.pad(img, padding)

        # shift image according to center of mass
        # Calculate the center of mass for each channel
        center_of_mass_rows = torch.sum(torch.arange(img.size(1)).float().view(1, -1, 1) * img,
                                        dim=(1, 2)) / torch.sum(img, dim=(1, 2))
        center_of_mass_cols = torch.sum(torch.arange(img.size(2)).float().view(1, 1, -1) * img,
                                        dim=(1, 2)) / torch.sum(img, dim=(1, 2))

        # Calculate the shift needed to center the image
        shift_rows = img.size(1) // 2 - center_of_mass_rows.view(-1, 1, 1)
        shift_cols = img.size(2) // 2 - center_of_mass_cols.view(-1, 1, 1)

        # Shift the image to center it
        img = torch.roll(img, shifts=(shift_rows.round().int().item(), shift_cols.round().int().item()),
                         dims=(1, 2))

        img = v2.functional.normalize_image(img, [0.1307, ], [0.3081, ])

        img = img.float().unsqueeze(axis=0)
        return img

