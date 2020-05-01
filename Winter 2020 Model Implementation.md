# Winter 2020 Model Implementation


## Intro

Throughout this term, two main types of models were focused on: ResNet-18 and GAN. The goal for ResNet-18 was to optimize its performance so that its classification results were comparable to those of fiTQun. For the GAN model, its main purpose was as a stepping stone to CycleGANs, with the eventual goal of mitigating systematic uncertainties in simulated data.

## Table of Contents

### 1. [ResNet-18](#resnet)
### 2. [GAN](#gan)

## ResNet-18 <a id="resnet"></a>


The architecture for the ResNet-18 model optimized during this past co-op term is made up of a ResNet-18 encoder followed by a latent classifier and is created using several scripts in the `models` folder. The derived class for this network is ClNet (in `clnet.py`). The base class `BaseModel` is found in `basemodel.py` and determines the encoder as specified by the config file - which will be edresnet for this model (`arch_key=1`). The latent classifier is instantiated in ClNet, and its architecture specified in the file `bottlenecks.py`.
The forward function of ClNet sends the data first through the encoder and then the latent classifier.

Trained model pathways:
The pathway for the best trained ResNet-18 model is `dumps/20200306_113956`. The test run for this model on a subset of the test set is found at `dumps/20200418_095532`, and the test run for the model on all test events (with the root files and eventids included) is at `dumps/20200308_195054`.

To run the ResNet-18 model, use `python watchmal_cl.py --load <config file>`. The config file should be in `config/engine_config` - I usualy used `test.ini`.


## GAN <a id="gan"></a>

The architecture for the GAN model begun this term uses a DCGAN architecture based on the [pytorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). The model is created and run using several scripts in the “models” folder. The class for this network is GanNet (in gannet.py). From here, the instatiator is called for the network’s generator and discriminator architecture. The architecture for the generator and discriminator is specified in “GeneratorDiscriminator.py”.
The forward function for GanNet is not used - instead, the forward pass in the GAN engine directly passes the data to forward functions of the generator and discriminator.

The first stepping stone in running a GAN model was to actually run the DCGAN tutorial. The results of this model run can be found under `dumps/20200417_225836`. Then, to adjust the architecture to fit our data, channel sizes were tweaked and the TanH activation removed (to reduce the need for data normalization).

To run the DCGAN model, use `python watchmal_gan.py --load <config file>`. The config file should be in `config/engine_config` - I usualy used `test.ini`.

```bash

```
