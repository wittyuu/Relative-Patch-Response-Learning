"""Single-image inference with a trained PRL detector.

    python tools/infer.py --weights prl_dinov3_vitl16.safetensors \
        --dinov3 /path/to/dinov3-vitl16-pretrain-lvd1689m --image a.png [b.jpg ...]

Evaluation protocol: in GenImage, UnivFD, ForenSynths and AIGCDetectBenchmark
the real images are JPEG files while most synthetic images are PNG, so the
synthetic images of these benchmarks are JPEG-compressed at quality 96 before
evaluation to remove the format bias. This script applies that compression to
files whose name starts with genimage_fake, univfd_fake, forensynths_fake or
aigcdetect_fake (case-insensitive); every other image is used as it is.
"""

import argparse
import os

import torch
from PIL import Image

from prl.data import jpeg_compress, preprocess
from prl.models import PRLDetector

JPEG_DEBIAS_QUALITY = 96
JPEG_DEBIAS_PREFIXES = (
    "genimage_fake",
    "univfd_fake",
    "forensynths_fake",
    "aigcdetect_fake",
)


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
    if os.path.basename(path).lower().startswith(JPEG_DEBIAS_PREFIXES):
        image = jpeg_compress(image, JPEG_DEBIAS_QUALITY)
    return preprocess(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True, help="PRL .safetensors weights")
    parser.add_argument(
        "--dinov3", required=True, help="Hugging Face id or local directory of DINOv3"
    )
    parser.add_argument("--image", required=True, nargs="+", help="image path(s)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = PRLDetector(args.dinov3)
    model.load_trainable_weights(args.weights)
    model.to(args.device).eval()
    

    with torch.no_grad():
        for path in args.image:
            logit, _ = model(load_image(path).unsqueeze(0).to(args.device))
            probability = torch.sigmoid(logit).item()
            label = "fake" if probability >= 0.5 else "real"
            print(f"{probability:.4f}  {label}  {path}")


if __name__ == "__main__":
    main()
