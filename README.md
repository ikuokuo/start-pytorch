# Start [PyTorch][]

[PyTorch]: https://pytorch.org/

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

## Research

- [Detectron2](research/detectron2/README.md)
