from io import BytesIO

import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def jpeg_compress(image, quality):
    if quality == 100:
        return image
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    output = Image.open(buffer).convert("RGB")
    output.load()
    return output


def preprocess(image, size=336, max_side=1024):
    width, height = image.size
    if width > max_side and height > max_side:
        scale = size / min(width, height)
        image = image.resize(
            (
                max(size, max(1, int(round(width * scale)))),
                max(size, max(1, int(round(height * scale)))),
            ),
            resample=BICUBIC,
        )
        width, height = image.size
    missing_width = max(0, size - width)
    missing_height = max(0, size - height)
    padding = (
        missing_width // 2,
        missing_height // 2,
        missing_width - missing_width // 2,
        missing_height - missing_height // 2,
    )
    if any(padding):
        image = TF.pad(image, list(padding), fill=0, padding_mode="constant")
    image = TF.center_crop(image, [size, size])
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
