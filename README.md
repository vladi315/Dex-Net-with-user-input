This repo is a fork of the Berkeley AUTOLAB's dex-net GQ-CNN. For their documentation and code see:

<https://berkeleyautomation.github.io/dex-net/>

<https://berkeleyautomation.github.io/gqcnn/>

<https://github.com/BerkeleyAutomation/gqcnn>

## Prerequisites

### Python

The gqcnn package has only been tested with Python 3.5, Python 3.6, and Python 3.7.

### Ubuntu

The gqcnn package has only been tested with Ubuntu 12.04, Ubuntu 14.04 and Ubuntu 16.04.

### Virtualenv

We highly recommend using a Python environment management system, in particular Virtualenv, with the Pip and ROS installations. Note: Several users have encountered problems with dependencies when using Conda.

    virtualenv -p /usr/bin/python3.7 ~/virtualenv/dex-net
    source ~/virtualenv/dex-net/bin/activate

## Pip Installation

The pip installation is intended for users who are only interested in 1) Training GQ-CNNs or 2) Grasp planning on saved RGBD images, not interfacing with a physical robot. If you have intentions of using GQ-CNNs for grasp planning on a physical robot, we suggest you install as a ROS package.

### 1. Clone the repository

Clone or download the project from Github.

    git clone git@code.siemens.com:FMP_Analytics/edgeapps-for-shop4cf/dex-net.git

### 2. Run pip installation

Change directories into the gqcnn repository and run the pip installation.

    pip install .

This will install gqcnn in your current virtual environment.

## Inference

With the virtualenv activated, run from the gqcnn directory execute:

    ./run_DexNet_with_user_input_example.sh

You can adjust the parameters defined in the shell script to point to your own data.

Note that ``segmask``, ``config_filename`` and ``user_input_fusion_method`` are optional parameters.

If ``user_input_fusion`` method is provided, also ``camera_pose_path``,  ``user_input_3d_dir`` must be provided

## Reference

@article{mahler2019learning,
  title={Learning ambidextrous robot grasping policies},
  author={Mahler, Jeffrey and Matl, Matthew and Satish, Vishal and Danielczuk, Michael and DeRose, Bill and McKinley, Stephen and Goldberg, Ken},
  journal={Science Robotics},
  volume={4},
  number={26},
  pages={eaau4984},
  year={2019},
  publisher={AAAS}
}
