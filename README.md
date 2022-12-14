# gradient-sdf-tracking
A new python code for the tracking part of the paper 

 **[Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction](https://arxiv.org/abs/2111.13652)**

 Our official C++ implementation is [here](https://github.com/c-sommer/gradient-sdf).


## Installation Requirement
The code is compatible with python 3.7 and pytorch 1.2. In addition, the following packages are required:
numpy, pyhocon, scikit-image, imageio, opencv.

You can create an anaconda environment called `grad-sdf` with the required dependencies by running:

```
conda env create -f environment.yml
conda activate grad-sdf
```

## Run the code
The main code to run is [tracker_main.py](code/main/tracker_main.py), which takes 3 input arguements:

- --conf: path the configuration file
- --results_dir: where you want to save results (optional)
- --expname: your current experiment 

Take `freiburg1_desk` data in [TUM-RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset) as an example:
```
cd code
python main/tracker_main.py --conf conf/tum.conf --expname desk
```

## Usage
The code need *RGBD data sequence* as input, together with camera intrinsics. For example, [TUM-RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset), [ETH3D](https://www.eth3d.net/), [Redwood](http://redwood-data.org/). Please specify the data format in the configuration file. 

The code deal with two data formats specified in configuration file `dataset: mode` `tum`: like [TUM-RGBD](code/conf/tum.confconf/), `other` [bunny](code/conf/bunny.conf) 

`tum`: deal with the data which stores rgb file names and depth file names in a txt list (associate.txt)

`other`: deal with the data that rgb and depth file names save in continuous number with a uniform prefix. For example, if your dataset has structure like 
```
-data
|-depth000.png
|-depth001.png
|-...
|-rgb000.png
|-rgb001.png
|-...
```
then in the `data.conf` file, dataset mode is `other`, depth_prefix is `depth`, `rgb_prefix` is `rgb`, `digital` is `3`, `first` is 0. 

### required config
in dataset
- data_path: dataset folder location
- mode: as explained
- assoc_file/prefix: if the mode is tum
- depth_prefix/rgb_prefix/digital: if the mode is other 
- intrinsics_file: path to intrinsics txt file
- depth factor: the scale that need to be *divided* after read depth map as image. Note this variable will be divide, not multiplied.

in model:
- img_res: image size


### Trouble shooting and to do

I noticed some performance differences between our official [c++ version](https://github.com/c-sommer/gradient-sdf) and this re-implemented python version. Please use our c++ version if you can.

C++ Version
- we have our own MarchingCubes which enables color mesh output
- use `Sophus` and `Eigen` package to deal with SE3 computation and solve linear systems
- stable tracking performence
- has PS optimization part

Python Version:
- no color mesh export
- self implemented SE3 computation might be the reason of unstable performance
- no PS refinement part (unfortunately I will not add this part)

TODO:
- [x] will add cuda version
- [x] will improve the mesh generating
- [ ] if you know some official python package for SO3 and SE3 calculation, please let me know :P

### Citation
If you find our work useful in your research, please consider citing:

```
@string{cvpr="IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"}
@inproceedings{Sommer2022,
 author = {C Sommer and L Sang and D Schubert and D Cremers},
 title = {Gradient-{SDF}: {A} Semi-Implicit Surface Representation for 3D Reconstruction},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2022},
 url = {https://arxiv.org/abs/2111.13652},
 titleurl = {sommer2022.png},
}

```

