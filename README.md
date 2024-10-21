<h1 align="center">Score-based Diffusion Models with Self-supervised Learning for Accelerated 3D Multi-contrast Cardiac MR Imaging</h1>

<p align="center">

</p>

    
Official code for the paper "Score-based Diffusion Models with Self-supervised Learning for Accelerated 3D Multi-contrast Cardiac MR Imaging"

## Abstract <a name = "Abstract"></a>

Long scan time significantly hinders the widespread applications of three-dimensional multi-contrast cardiac magnetic resonance (3D-MC-CMR) imaging. This study aims to accelerate 3D-MC-CMR acquisition by a novel method based on score-based diffusion models with self-supervised learning. Specifically, we first establish a mapping between the undersampled k-space measurements and the MR images, utilizing a self-supervised Bayesian reconstruction network. Secondly, we develop a joint score-based diffusion model on 3D-MC-CMR images to capture their inherent distribution. The 3D-MC-CMR images are finally reconstructed using the conditioned Langenvin Markov chain Monte Carlo sampling. This approach enables accurate reconstruction without fully sampled training data. Its performance was tested on the dataset acquired by a 3D joint myocardial T1 and T1ρ mapping sequence. The T1 and T1ρ maps were estimated via a dictionary matching method from the reconstructed images. Experimental results show that the proposed method outperforms traditional compressed sensing and existing self-supervised deep learning MRI reconstruction methods. It also achieves high quality T1 and T1ρ parametric maps close to the reference maps obtained by traditional mapping sequences, even at a high acceleration rate of 14.

## Setup 
The following will introduce environment setup, data preparation, usage instructions, experiment weights, and a quick start guide. In the quick start guide, we provide experiment weights and synthetic data for easy model validation without the need for lengthy data downloads. Additionally, the quick start guide will offer guidance on parameter settings for generating high-quality reconstruction samples.

### Dependencies
Run the following to install a subset of necessary python packages for our code.

```sh
pip env create -f environment.yml
```

### Data Preparation
To facilitate a quick start, we provide sample data located at data/train.

### Trained Models
After setting up the environment and downloading dataset files, you can simply run the following commands.
