<div align="center">
<h1>Relative Patch Response Learning for Generalizable AI-Generated Image Detection</h1>
</div>

This repository is the official PyTorch implementation of **PRL** (Relative
Patch Response Learning). PRL trains an AI-generated image detector on the
*patch responses* of aligned real/synthetic image pairs: how the score of each
patch changes between two mixed views of the same pair.

---

## ⚙️ Installation

```bash
conda create -n prl python=3.10 -y && conda activate prl
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Inference

```bash
python tools/infer.py --weights weights/prl_dinov3_vitl16.safetensors \
    --dinov3 /path/to/dinov3-vitl16 \
    --image path/to/image.png [more images ...]
```

> **JPEG-96 protocol.** In GenImage, UnivFD, ForenSynths and
> AIGCDetectBenchmark the real images are JPEG files while most synthetic images
> are not. Following the evaluation protocol, the synthetic images of these four
> benchmarks are JPEG-compressed at quality 96 before inference.

---

## 📑 Checkpoint

| Model | Backbone | Size | File |
|---|---|---|---|
| PRL | DINOv3 ViT-L/16 | 16 MB | [`weights/prl_dinov3_vitl16.safetensors`](weights/prl_dinov3_vitl16.safetensors) |

The pretrained DINOv3 ViT-L/16 weights need to be prepared separately and
passed to `--dinov3`.

---

## 🎯 ToDo List

- [x] Release checkpoint and inference code
- [ ] Release training code

---

## 😄 Acknowledgement

The LoRA module is adapted from
[Dual-Data-Alignment](https://github.com/roy-ch/Dual-Data-Alignment), and the
backbone is [DINOv3](https://github.com/facebookresearch/dinov3), whose weights
are subject to the DINOv3 License. Huge thanks to the authors for sharing their
excellent work!

---

## ✍️ Citing

*To be added.*

---

## 📄 License

This code is released under the [Apache-2.0 License](LICENSE).
