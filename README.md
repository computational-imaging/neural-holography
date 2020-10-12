# Neural Holography with Camera-in-the-loop Training
### [Project Page](http://www.computationalimaging.org/publications/neuralholography/)  | [Paper](http://www.computationalimaging.org/wp-content/uploads/2020/08/NeuralHolography_SIGAsia2020.pdf)

[Yifan Peng](http://stanford.edu/~evanpeng/), [Suyeon Choi](https://choisuyeon.github.io/), [Nitish Padmanaban](https://nitish.me/), [Gordon Wetzstein](http://stanford.edu/~gordonwz/)

This repository contains the scripts associated with the SIGGRAPH Asia 2020 paper "Neural Holography with Camera-in-the-loop Training"

> The camera-in-the-loop optimization and the parameterized wave propagation model / Holonet training code will be released during SIGGRAPH Asia 2020 (Dec. 2020)
## Getting Started

You can set up a conda environment with all dependencies like so (for Linux):
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
To run phase generation with Holonet/U-net, download the pretrained model weights from [here](https://drive.google.com/file/d/1Xr353I3ycRFBXLoIjYTzUbdWzurW0N_H/view?usp=sharing) and place the contents in the ```pretrained_networks/``` folder. 

## High-level structure

The code is organized as follows:

* ```main.py``` generates phase patterns via SGD/GS/DPAC/Holonet/U-net.
* ```eval.py``` reconstructs and evaluates with optimized phase patterns. 
* ```main_eval.sh``` first executes ```main.py``` for RGB channels and then executes ```eval.py```



* ```propagation_ASM.py``` contains the wave propagation operator (angular spectrum method)
* ```holenet.py``` contains modules of HoloNet/U-net implementations.
* ```algorithms.py``` contains GS/SGD/DPAC algorithm implementations.

./utils/
* ```utils.py``` contains utility functions.
* ```modules.py``` contatins PyTorch wrapper modules for easy use of ```algorithms.py```.
* ```pytorch_prototyping/``` submodule contains custom pytorch modules with sane default parameters. (adapted from [here](https://github.com/vsitzmann/pytorch_prototyping))
* ```augmented_image_loader.py``` contains modules of loading a set of images.


## Running the test

You can simply executes the following bash script with a method parameter (You can replace the parameter ```SGD``` with ```GS/DPAC/HOLONET/UNET```.):
```
chmod u+x main_eval.sh
./main_eval.sh SGD
```


This bash script executes the phase optimization with 1) ```main.py``` for each R/G/B channel, and then executes 2) ```eval.py```, which simulates the holographic image reconstruction for the optimized patterns with the angular spectrum method.
Check ```./phases``` and ```./recon``` folders after the execution.

### 1) Phase optimization
The SLM phase patterns can be reproduced with

SGD (Gradient Descent):
```
python main.py --channel=0 --method=SGD --root_path=./phases
```
HoloNet
```
python main.py --channel=0 --method=HOLONET --root_path=./phases --model_dir=./pretrained_networks
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
python main.py --channel=0 --method=UNET --root_path=./phases --model_dir=./pretrained_networks
```
You can set ```--channel=1/2``` for other (green/blue) channels.

To monitor progress, the optimization code writes tensorboard summaries into a "summaries" subdirectory in the ```root_path```.

### 2) Simulation
With optimized phase patterns, you can simulate the holographic image reconstruction with 

```
python eval.py --channel=0 --root_path=./phases/SGD_ASM
```

For full-color simulation, you can set ```--channel=3```, ```0/1/2``` corresponds to `R/G/B`, respectively.

This simulation code writes the reconstruction images in ```./recon``` folder as default.

Feel free test other images after putting them in ```./data``` folder!


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