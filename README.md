# Start [PyTorch][]

[PyTorch]: https://pytorch.org/
[PyTorch Examples]: https://github.com/pytorch/examples

## Environment

### Ubuntu

- https://wiki.ubuntu.com/Releases
- Ubuntu 18.04.5 LTS (Bionic Beaver)
  - ubuntu-18.04.5-desktop-amd64.iso
  - http://releases.ubuntu.com/bionic/

### Anaconda

- https://www.anaconda.com/distribution/
- Anaconda Python 3.8
  - Anaconda3-2020.11-Linux-x86_64.sh

### PyTorch

- [Get Started](https://pytorch.org/get-started/)
- [Support Versions](https://github.com/pytorch/vision#installation)

#### Stable

```bash
# NOTE: Python 3.9 users will need to add '-c=conda-forge' for installation
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch -y
```

#### Nightly

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly -y
```

## Preparation

```bash
conda activate pytorch

git clone https://github.com/ikuokuo/start-pytorch.git

cd start-pytorch
pip install -r requirements.txt
```

## Tutorials

- [PyTorch Custom Dataset](docs/torch/torch_custom_dataset.md)

torchvision:

- [TorchVision Inference with a Pretrained Model](docs/torchvision/torchvision_inference_with_a_pretrained_model.md)
- [TorchVision Object Detection Finetuning Tutorial](docs/torchvision/finetuning_object_detection/torchvision_finetuning_object_detection.ipynb)
  - Kaggle notebooks about this tutorial 👇
    - [TorchVision Faster R-CNN Finetuning](https://www.kaggle.com/gocoding/torchvision-faster-r-cnn-finetuning)
    - [TorchVision Faster R-CNN Inference](https://www.kaggle.com/gocoding/torchvision-faster-r-cnn-inference)
- [TorchVision Instance Segmentation Finetuning Tutorial](docs/torchvision/finetuning_instance_segmentation/torchvision_finetuning_instance_segmentation.ipynb)

others:

- [Use Kaggle Notebooks](docs/use_kaggle_notebooks.md)
