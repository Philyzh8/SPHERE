# SPHERE

[![PyPI - Python Version](https://img.shields.io/pypi/v/stDCL)](https://pypi.org/project/stDCL/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`SPHERE` is a Python package containing tools for identifing spatial domains from spatial transcriptomics data based on a dual graph contrastive learning method.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [License](#license)


# Overview
Spatial epigenomic technologies offer unprecedented opportunities to profile regulatory landscapes by jointly resolving chromatin accessibility and tissue architecture. However, spatial epigenomic measurements are often fragmented, typically limited to sparse serial sections and uneven, sparse and partially overlapping sampling across epigenomic platforms, developmental stages, instrument resolutions and fields of view (FOVs); it obscures regulatory continuity and precludes unified interpretation. We formulate this challenge as a panoramic spatiotemporal stitching (PaSS) problem: reconstructing a unified, continuous regulatory landscape when measurements are fragmented, partially overlapping, and confounded by platform-specific artifacts. Here, we present SPHERE (Spatial Panoramic Holistic Epigenomic REpresentation), a unified framework that models each slice with spatial and regulatory graphs, fuses them via attention, and enforces cross-slice manifold consistency to mitigate platform-specific biases while preserving developmental progression. Benchmarking across simulations and multiple real datasets demonstrates that SPHERE maintains platform-agnostic structural consistency and developmental continuity under cross-platform and spatiotemporal fragmentation. Leveraging eleven mouse embryonic tissue slices profiled by four spatial epigenomic platforms, we perform omni-platform spatiotemporal stitching to construct a unified atlas spanning six developmental stages, revealing fine-grained regulatory programs and trajectory dynamics in the developing brain. This unified latent manifold simultaneously learns the biological and physical dimensions of PaSS, while cell-type resolved panoramic annotation ensures high-resolution insights and precise panoramic reconstruction further unfolds into a coherent 3D volumetric landscape that recapitulates complex brain structures. By stitching sparsely sampled, partially overlapping spatial epigenomic sections into an integrated atlas, SPHERE democratizes omni-platform spatiotemporal analysis of tissue regulation without requiring fully crossed measurements.

<img width="1454" height="1566" alt="image" src="https://github.com/user-attachments/assets/14219059-beff-4919-be64-453af67d9da2" />


# System Requirements
## Hardware requirements
`SPHERE` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 22.04

### Python Dependencies
`SPHERE` mainly depends on the Python scientific stack.
```
numpy
scipy
torch
scikit-learn
pandas
scanpy
```
For specific setting, please see <a href="https://github.com/Philyzh8/SPHERE/blob/master/requirements.txt">requirements</a> or <a href="https://github.com/Philyzh8/SPHERE/blob/master/environment.yaml">environment</a>.

# Installation Guide:

### Install from PyPi

```
$ conda create -n SPHERE_env python=3.8.15
$ conda activate SPHERE_env
$ pip install -r requirements.txt
$ pip install SPHERE
```

### Install from Conda

```
$ conda env create -f environment.yaml
```

# Usage
:page_facing_up: `SPHERE` is a unified framework that models each slice with spatial and regulatory graphs, fuses them via attention, and enforces cross-slice manifold consistency to mitigate platform-specific biases while preserving developmental progression, which can be used to:
+ <a href="https://github.com/Philyzh8/SPHERE/tree/master/Tutorial/Tutorial1%3A%20platform-agnostic%20structural%20consistency">Tutorial1</a>. platform-agnostic structural consistency.
+ <a href="https://github.com/Philyzh8/SPHERE/tree/master/Tutorial/Tutorial2%3A%20spatiotemporal%20integration">Tutorial2</a>. spatiotemporal integration.
+ <a href="https://github.com/Philyzh8/SPHERE/tree/master/Tutorial/Tutorial3%3A%20omni-platform%20spatiotemporal%20integration">Tutorial3</a>. omni-platform spatiotemporal integration.
+ <a href="https://github.com/Philyzh8/SPHERE/tree/master/Tutorial/Tutorial4%3A%20panoramic%20annotation">Tutorial4</a>. panoramic annotation.
+ <a href="https://github.com/Philyzh8/SPHERE/tree/master/Tutorial/Tutorial5%3A%20panoramic%20reconstruction">Tutorial5</a>. panoramic reconstruction.


# License

This project is covered under the <a href="https://github.com/Philyzh8/SPHERE/blob/master/LICENSE">**MIT License**</a>.


