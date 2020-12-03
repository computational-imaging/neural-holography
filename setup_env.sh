source ~/anaconda3/etc/profile.d/conda.sh
conda create -n neural-holography python=3.6 scipy opencv
conda activate neural-holography
conda install -c conda-forge opencv
conda install pytorch torchvision -c pytorch
conda install -c conda-forge tensorboard
conda install -c anaconda scikit-image
pip install ConfigArgParse
conda install -c conda-forge opencv
pip install aotools

