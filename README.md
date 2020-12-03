# Neural Holography with Camera-in-the-loop Training
### [Project Page](http://www.computationalimaging.org/publications/neuralholography/)  | [Paper](http://www.computationalimaging.org/wp-content/uploads/2020/08/NeuralHolography_SIGAsia2020.pdf)

[Yifan Peng](http://stanford.edu/~evanpeng/), [Suyeon Choi](https://choisuyeon.github.io/), [Nitish Padmanaban](https://nitish.me/), [Gordon Wetzstein](http://stanford.edu/~gordonwz/)

This repository contains the scripts associated with the SIGGRAPH Asia 2020 paper "Neural Holography with Camera-in-the-loop Training"

## Update 20201203: 
We just released the second part of our scripts.

It contains all the code that can reproduce our work, including the camera-in-the-loop optimization, the parameterized wave propagation model / Holonet training code.
Also, we are publishing our hardware SDK incorporation code for automated pipelines. Please have a look and feel free to modify and use for your work!

The specific updates can be found in following sections:
- [2.2) CITL-calibrated model simulation](#22-citl-calibrated-model-simulation), 
- [2.3) Evaluation on the physical setup](#23-evaluation-on-the-physical-setup), 
- [3) Training](#3-training), 
- [4) Hardwares](#4-hardwares-camera-slm-laser-automation-and-calibration)

## Getting Started

**Our code requires PyTorch >1.7.0, as it uses Complex64 type Tensors.**
You can implement it in previous versions of PyTorch with the complex number operations implemented in ```utils/utils.py```.

You can set up a conda environment with all dependencies like so:

For Windows:
```
conda env create -f environment_windows.yml
conda activate neural-holography
```

For Linux: (Hardware SDKs may not be compatible)
```
conda env create -f environment.yml
conda activate neural-holography
```
or you can manually set up a conda environment with (just execute ```setup_env.sh``` if you use Windows):
```
chmod u+x setup_env.sh
./setup_env.sh
conda activate neural-holography
```

You can load the [submodule](https://github.com/vsitzmann/pytorch_prototyping) in ```utils/pytorch_prototyping``` folder with
```
git submodule init
git submodule update
```
To run phase generation with Holonet/U-net, download the pretrained model weights from [here](https://drive.google.com/file/d/1Xr353I3ycRFBXLoIjYTzUbdWzurW0N_H/view?usp=sharing) and place the contents in the ```pretrained_networks/``` folder. 

To run Camera-in-the-loop optimization or training, download [PyCapture2 SDK](https://www.flir.com/products/flycapture-sdk/) and [HOLOEYE SDK](https://holoeye.com/spatial-light-modulators/slm-software/slm-display-sdk/) and place the SDKs in your environment folder. If your hardware setup (SLM, Camera, Laser) is different from ours, please modify related files in the ```utils/``` folder according to the SDK before running the camera-in-th-loop optimization or training.
Our hardware specifications can be found in the [paper](http://www.computationalimaging.org/wp-content/uploads/2020/08/NeuralHolography_SIGAsia2020.pdf) (Appendix B).


## High-level structure

The code is organized as follows:

* ```main.py``` generates phase patterns via SGD/GS/DPAC/Holonet/U-net.
* ```eval.py``` reconstructs and evaluates with optimized phase patterns. 
* ```main_eval.sh``` first executes ```main.py``` for RGB channels and then executes ```eval.py```.



* ```propagation_ASM.py``` contains the wave propagation operator (angular spectrum method).
* ```propagation_model.py``` contains our parameterized wave propagation model.
* ```holenet.py``` contains modules of HoloNet/U-net implementations.
* ```algorithms.py``` contains GS/SGD/DPAC algorithm implementations.



* ```train_holonet.py``` trains Holonet with ASM or the CITL-calibrated model.
* ```train_model.py``` trains our wave propagation model with camera-in-the-loop training.


./utils/
* ```utils.py``` contains utility functions.
* ```modules.py``` contatins PyTorch wrapper modules for easy use of ```algorithms.py``` and our hardware controller.
* ```pytorch_prototyping/``` submodule contains custom pytorch modules with sane default parameters. (adapted from [here](https://github.com/vsitzmann/pytorch_prototyping))
* ```augmented_image_loader.py``` contains modules of loading a set of images.



* ```utils_tensorboard.py ``` contains utility functions used for visualization on tensorboard.
* ```slm_display_module.py ``` contains the SLM display controller module. ([HOLOEYE SDK](https://holoeye.com/spatial-light-modulators/slm-software/slm-display-sdk/))
* ```detect_heds_module_path.py``` sets the SLM SDK path. Otherwise you can copy the holoeye module directory into your project and import by using ```import holoeye```.
* ```camera_capture_module.py ``` contains the FLIR camera capture controller module. ([PyCapture2 SDK](https://www.flir.com/products/flycapture-sdk/))
* ```calibration_module.py ``` contains the homography calibration module.

## Running the test

You can simply execute the following bash script with a method parameter (You can replace the parameter ```SGD``` with ```GS/DPAC/HOLONET/UNET```.):
```
chmod u+x main_eval.sh
./main_eval.sh SGD
```


This bash script executes the phase optimization with 1) ```main.py``` for each R/G/B channel, and then executes 2) ```eval.py```, which simulates the holographic image reconstruction for the optimized patterns with the angular spectrum method.
Check the ```./phases``` and ```./recon``` folders after the execution.

### 1) Phase optimization
The SLM phase patterns can be reproduced with

SGD (Gradient Descent):
```
python main.py --channel=0 --method=SGD --root_path=./phases
```

SGD with Camera-in-the-loop optimization:
```
python main.py --channel=0 --method=SGD --citl=True --root_path=./phases
```

SGD with CITL-calibrated models:
```
python main.py --channel=0 --method=SGD --prop_model='MODEL' --prop_model_dir=YOUR_MODEL_PATH --root_path=./phases
```
HoloNet
```
python main.py --channel=0 --method=HOLONET --root_path=./phases --generator_dir=./pretrained_networks
```

GS (Gerchberg-Saxton):
```
python main.py --channel=0 --method=GS --root_path=./phases
```

DPAC (Double Phase Encoding):
```
python main.py --channel=0 --method=DPAC --root_path=./phases
```
U-net
```
python main.py --channel=0 --method=UNET --root_path=./phases --generator_dir=./pretrained_networks
```
You can set ```--channel=1/2``` for other (green/blue) channels.

To monitor progress, the optimization code writes tensorboard summaries into a "summaries" subdirectory in the ```root_path```.

### 2) Simulation/Evaluation
#### 2.1) Ideal model simulation: 

With optimized phase patterns, you can simulate the holographic image reconstruction with 

```
python eval.py --channel=0 --root_path=./phases/SGD_ASM --prop_model=ASM
```

For full-color simulation, you can set ```--channel=3```, ```0/1/2``` corresponds to `R/G/B`, respectively.

This simulation code writes the reconstruction images in ```./recon``` folder as default.

Feel free test other images after putting them in ```./data``` folder!

#### 2.2) CITL-calibrated model simulation

You can simulate those patterns with CITL-calibrated model

```
python eval.py --channel=0 --root_path=./phases/SGD_ASM --prop_model=MODEL
```

#### 2.3) Evaluation on the physical setup

You can capture the image on the physical setup with

```
python eval.py --channel=0 --root_path=./phases/SGD_ASM --prop_model=CAMERA
```

### 3) Training
There are two-types of training in our work: 
1) Parameterized wave propagation model (Camera-in-the-loop training)
2) Holonet.

Note that we need the camera-in-the-loop for training the wave propagation model while the Holonet can be trained offline once the CITL-calibrated model is calibrated.
(You can pre-capture a bunch of phase-captured image pairs and can train the wave propagation models offline as well, though.)

You can train our wave propagation models with

```
python train_model.py --channel=0
```

You can train Holonet with

```
python train_holonet.py  --perfect_prop_model=True --run_id=my_first_holonet --batch_size=4 --channel=0
```

If you want to train it with CITL-calibrated models, set ```perfect_prop_model``` option to ```False``` and set ```model_path``` to your calibrated models.
 
### 4) Hardware (Camera, SLM, laser) Automation and Calibration
We incorporated the hardware SDKs as a pytorch module so that we can easily capture the experimental results and put them **IN-THE-LOOP**. 

You can call the module with
```
camera_prop = PhysicalProp(channel, roi_res=YOUR_ROI,
                           range_row=(220, 1000), range_col=(300, 1630),
                           patterns_path=opt.calibration_path, # path of 21 x 12 calibration patterns, see Supplement.
                           show_preview=True)
```
Here, you may want to naively crop around the calibration patterns by setting ```range_row/col``` manually with the preview so that it can calculate the homography matrix without problems. (See Section S5 of [Supplement](https://drive.google.com/file/d/1vay4xeg5iC7y8CLWR6nQEWe3mjBuqCWB/view))

Then, you can get camera-captured images by simply sending SLM phase patterns through the forward pass of the module:

```
captured_amp = camera_prop(slm_phase)
```


## Citation
If you find our work useful in your research, please cite:

```
@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {Neural Holography with Camera-in-the-loop Training},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
issue = {39},
number = {6},
year = {2020},
}
```

## License
This project is licensed under the following license, with exception of the file "data/1.png", which is licensed under the [CC-BY](https://creativecommons.org/licenses/by/3.0/) license.


Copyright (c) 2020, Stanford University

All rights reserved.

Redistribution and use in source and binary forms for academic and other non-commercial purposes with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code, including modified source code, must retain the above copyright notice, this list of conditions and the following disclaimer.

* Redistributions in binary form or a modified form of the source code must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of The Leland Stanford Junior University, any of its trademarks, the names of its employees, nor contributors to the source code may be used to endorse or promote products derived from this software without specific prior written permission.

* Where a modified version of the source code is redistributed publicly in source or binary forms, the modified source code must be published in a freely accessible manner, or otherwise redistributed at no charge to anyone requesting a copy of the modified source code, subject to the same terms as this agreement.

THIS SOFTWARE IS PROVIDED BY THE TRUSTEES OF THE LELAND STANFORD JUNIOR UNIVERSITY "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE LELAND STANFORD JUNIOR UNIVERSITY OR ITS TRUSTEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Contact
If you have any questions, please contact

* Yifan (Evan) Peng, evanpeng@stanford.edu
* Suyeon Choi, suyeon@stanford.edu 
* Gordon Wetzstein, gordon.wetzstein@stanford.edu 