# NeSyXIL (Neuro-Symbolic Explanatory Interactive Learning)

This is the official repository of the article: [Right for the Right Concept: Revising Neuro-Symbolic Concepts by 
Interacting with their Explanations](https://arxiv.org/pdf/2011.12854.pdf) by Wolfgang Stammer, Patrick Schramowski, 
Kristian Kersting, to be published at CVPR 2021.

![Concept Learner with NeSy XIL](./figures/main_method.png)

This repository contains all source code required to reproduce the experiments of the paper including the ColorMNIST 
experiments and running the CNN model. In case you are only looking for the NeSy Concept Learner please visit the 
separate [repository](https://github.com/ml-research/NeSyConceptLearner).

Included is a docker file as well as a modified version of captum ([official repo](https://captum.ai/)) that we had 
modified by a single line such that the gradients of the explanations are kept. Simply running the docker file will 
automatically load the captum_xil version. (line 105 in ```NeSyXIL/src/docker/captum_xil/captum/attr/_utils/gradient.py```) 

## How to Run (in quick repititions move one leg in front of the other):

### CLEVR-Hans

#### Dataset

Please visit the [CLEVR-Hans](https://github.com/ml-research/CLEVR-Hans) repository for instrucitons on how to download 
the CLEVR-Hans dataset.

### NeSy Concept Learner

We have moved the source code for the NeSy Concept Learner to a separate 
[repository](https://github.com/ml-research/NeSyConceptLearner) in case someone is only interested in the Concept 
Learner for their work. In order to reproduce our experiments please clone this repository first:

1. ```cd src/clevr_hans/nesy_xil/```

2. ```git clone https://github.com/ml-research/NeSyConceptLearner.git```

### NeSy XIL experiments



### CNN experiments

We have included the trained CNN model files for seed 0. To reproduce the CNN experiments please runs the desired 
scripts e.g. as 

```cd src/clevr_hans/cnn/```

```./scripts/clevr-hans-resnet-xil.sh 0 0 /workspace/datasets/CLEVR-Hans3/```

For training with the HINTLoss on CLEVR-Hans3 with GPU 0 and run number 0.

### ColorMNIST

For running the ColorMNIST experiment with the provided dockerfile:

1. ```cd src/docker/```

2. ```docker build -t nesy-xil -f Dockerfile .```

3. ```docker run -it -v /home/ml-stammer/Documents/repositories/NeSyXIL:/workspace/repositories/NeSyXIL -v /home/ml-stammer/Documents/datasets/CLEVR-Hans3:/workspace/datasets/CLEVR-Hans3 --name nesy-xil --entrypoint='/bin/bash' --user $(id -u):$(id -g) --runtime nvidia nesy-xil```

4. ```cd src/color_mnist/data/```

5. ```python generate_color_mnist.py```

6. ```cd ..```

7. ```./scripts/color_mnist_train.sh 0``` for running on gpu 0

## Citation
If you find this code useful in your research, please consider citing:

> @article{stammer2020right,
  title={Right for the Right Concept: Revising Neuro-Symbolic Concepts by Interacting with their Explanations},
  author={Stammer, Wolfgang and Schramowski, Patrick and Kersting, Kristian},
  journal={arXiv preprint arXiv:2011.12854},
  year={2020}
}