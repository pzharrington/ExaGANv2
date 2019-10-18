# ExaGAN v2
**TF2.0 version of ExaGAN, with experimental spectral constraints and regularization**
![Image](https://drive.google.com/uc?export=view&id=1S-IBiylIACpPc25UUJXg5FsQ44IX9tO7)

---

See [ExaGAN repo](https://github.com/pzharrington/ExaGAN) for a more detailed description of the probem at hand. This repository builds on the same basic architecture as in ExaGAN, but with code re-written in TF2.0 to enable quick testing of various loss constraints and regularizers.

### Requirements
The code has been developed and run with the following software:
* numpy, scipy, matplotlib
* tensorflow 2.0.0-beta1
* tensorboard 1.13.1

### Data
See [ExaGAN repo](https://github.com/pzharrington/ExaGAN) for data download instructions. Simple slicing/data preparation script (copied from ExaGAN v1) included in `./data/` directory. 

Target mean and std for each `k` bin in the power spectrum `P(k)` have been pre-computed across dataset, and are saved in the `./data/` directory for use as a spectral constraint loss term for the generator. These will need to be recomputed for a new dataset of different samples/image dimensions.

### Code
To train, simpy do `python train.py 001`, which will create a unique directory `./expts/run001/` to store training checkpoints and tensorboard logs.

Model definition is in the `spectraGAN.py` file -- network architectures are identical to those of ExaGAN, but added features include the following:

* Spectral constraints -- penalizes generator when the generated samples have a power spectrum that differs from the mean power spectrum of the real data (see the ExaGAN repo for notes on the power spectrum). The key to enforcing this constraint is computing 'backpropagation-enabled' FFTs on the output of the generator, which tensorflow has support for (in 2D and 3D). The 2D FFT output is converted into a 1D power spectrum according to [this method](https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/).
* 'R1' regularization -- penalizes the discriminator gradients to remain small when processing the real data samples to avoid unstable updates to generator when close to the data manifold. See [Mescheder et al. 2018](https://arxiv.org/abs/1801.04406) for details.
* Feature matching -- penalizes the generator to match the statistics of the intermediate feature maps from the discriminator on the real data, which helps stabilize the training and prevent mode collapse. See [Salimans et al. 2016](https://arxiv.org/abs/1606.03498) for details.

At the moment, still working on idetnifying which of these are most necessary or important -- so far it is evident that the power spectrum constraints improve the average quality of the generator output, such that results are as good or better than those acquired using the MCR techinque (from ExaGAN v1 repository). Current implementation of power spectrum constraint is to penalize the generator based on the log MSE between the batchwise mean and variance of P(k) per k bin on the generated samples and the respective statistics of P(k) over the training set (the "target" distribution).


