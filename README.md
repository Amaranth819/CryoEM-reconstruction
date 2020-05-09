## Cryo-EM Reconstruction

  This repository is about synthesizing 3D Cryo-EM structures using deep learning. Now the network performs well on some structures (see the results under ./eval), but I will continue working on the refinement.

  Thanks for Professor Garrett Katz's instruction.



### Network

  The network is based on [3D-R2N2](https://arxiv.org/abs/1604.00449), a multi-view 3D reconstruction network.



### Environment

  My environment is Python 3.7.4 and Ubuntu 18.04. Also you need to install

  Before running the code, you need to install [RELION](https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Main_Page), which is used for synthesizing 2D projections. To visualize the predicted structures, download [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/).



### About the data

  You can use any real dataset or a synthetic dataset from [here](https://github.com/Amaranth819/CryoEM-Data-Synthesis).



### Usage

 1. Install all the requirements by

    `pip install -r requirements.py`

 2. Synthesize 2D projections by

    `./syn_2dproj.sh`

 3. Start the training by

    `python -u train.py`

 4. Visualize the loss by

    `tensorboard --logdir=/your/summary/path/`



### Warning

  When training with different structures, please modify the variable 'n_gru_vox' in res_net_gru.py. It depends on the input image size.




### Update on Feb 8
  Normalization on 2D projections and the 3D structure ground truth can improve the performance. I used the following [EMAN2](https://blake.bcm.edu/emanwiki/EMAN2) command line:

	1. `e2proc2d.py /your/2d/projection/stack/path /your/output/path --process normalize.edgemean`

 	2. `e2proc3d.py /your/3d/structure/path /your/output/path --process normalize.edgemean`



### Update on May 8

  You can get the synthetic training data from [here](https://github.com/Amaranth819/CryoEM-Data-Synthesis).