# Versatile Head Alignment with Adaptive Appearance Priors

<div align="center"> 
  <img src="asset/teaser.gif">
</div>

## TL;DR

- A photometric optimization pipeline based on differentiable rasterization, applied to human head alignment.
- A perturbation mechanism that implicitly extract and inject regional appearance priors adaptively during rendering.
- Enabling alignment of regions purely based on their appearance consistency, such as the hair, ears, neck, and shoulders, where no pre-defined landmarks are available.

## License

This work is made available under [CC-BY-NC-SA-4.0](./LICENSE). The repository is derived from the [multi-view head tracker of GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars/tree/main/reference_tracker), which is subjected to the following statements:

> Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to this software and related documentation. Any commercial use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.

On top of the original repository, we add support to monocular videos and provide a complete set of scripts from video preprocessing to result export for NeRF/3DGS-style applications.

## Setup

```shell
conda create --name VHAP -y python=3.10
conda activate VHAP

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake  # use the right CUDA version
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX  # for compilation

# Install PyTorch (make sure that the CUDA version matches with "Step 1")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# or
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
# make sure torch.cuda.is_available() returns True

pip install -e .
```

## Note

- We use an adjusted version of [nvdiffrast](https://github.com/ShenhanQian/nvdiffrast/tree/backface-culling) for backface-culling. To completely remove previous versions and compiled pytorch extensions, you can execute

  ```shell
  pip uninstall nvdiffrast
  rm -r ~/.cache/torch_extensions/*/nvdiffrast*
  ```

- We use [STAR](https://github.com/ShenhanQian/STAR/) for landmark detection by default. Alterntively, [face-alignment](https://github.com/1adrianb/face-alignment) is faster but less accurate.

## Download

### FLAME

Our code relies on FLAME. Downloaded asset from the [official website](https://flame.is.tue.mpg.de/download.php) and store them in the paths below:

- `asset/flame/flame2023.pkl`  # FLAME 2023 (versions w/ jaw rotation)
- `asset/flame/FLAME_masks.pkl`  # FLAME Vertex Masks

> NOTE: It is possible to use FLAME 2020 by download to `asset/flame/generic_model.pkl`. The `FLAME_MODEL_PATH` in `flame.py` needs to be updated accordingly.

### Video Data

#### Multiview

- To use [NeRSemble](https://tobias-kirschstein.github.io/nersemble/) dataset, please request via [Google Form](https://forms.gle/rYRoGNh2ed51TDWX9) to get approval and download links. You can find expected directory structure [here](https://github.com/ShenhanQian/VHAP/blob/c9ea660c6c6719110eca5ffdaf9029a2596cc5ca/vhap/data/nersemble_dataset.py#L32-L54).

#### Monocular

- We collect monocular video sequences from [INSTA](https://zielon.github.io/insta/). You can download raw videos from [LRZ](https://syncandshare.lrz.de/getlink/fiJE46wKrG6oTVZ16CUmMr/VHAP).

## Usage

### [For Monocular Videos](doc/monocular.md)

<div align="center"> 
  <img src="asset/monocular.jpg">
</div>

### [For NeRSemble Dataset](doc/nersemble.md)

<div align="center"> 
  <img src="asset/nersemble.jpg">
</div>

## Cite

Please kindly cite our repository and preceding paper if you find our software or algorithm useful for your research.

```bibtex
@article{qian2024versatile,
  title   = "Versatile Head Alignment with Adaptive Appearance Priors",
  author  = "Qian, Shenhan",
  year    = "2024",
  month   = "September",
  url     = "https://github.com/ShenhanQian/VHAP"
}
```

```bibtex
@inproceedings{qian2024gaussianavatars,
  title={Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians},
  author={Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20299--20309},
  year={2024}
}
```
