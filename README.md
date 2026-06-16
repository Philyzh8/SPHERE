# SPHERE

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`SPHERE` is a unified computational framework designed to address PAC for fragmented spatial epigenomes.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [License](#license)


# Overview
Spatial atlases are drawn as continuous maps but measured as scattered fragments that differ across platforms, stages, fields of view, resolutions and modalities. Integration reconciles the fragments that were observed, yet it cannot say which missing correspondences between them are supported by evidence, or where that inference can be trusted. We therefore recast atlas construction not as integration of observed fragments but as panoramic atlas completion (PAC), the task of inferring missing correspondences, stating when their completion is reliable, and realizing the result in physical tissue space. We introduce SPHERE, a unified and scalable framework that completes fragmented spatial epigenomic measurements into regulatory atlases across platforms, stages and resolutions. Most strikingly, when two sections share no physical overlap at all, SPHERE recovers coherent structure without shared physical landmarks, using regulatory topology where overlap-dependent integration fails. As a direct test of when completion can be trusted, SPHERE returns entirely held-out sections, including unseen developmental stages and sections from unseen platforms, to their correct positions within a pre-trained atlas, making completion a measurable quantity rather than an assumption. Rather than report aggregate scores alone, we map the operating regime of PAC, showing where completion stays reliable and where it does not. Across eleven mouse embryonic sections from four platforms and six stages, together with human cerebellum and thymus data, SPHERE recovers anatomically faithful structures, resolves telencephalon trajectories during corticogenesis, scales to multi-million high-resolution bins and assembles coherent 3D volumetric landscapes. It meets two requirements that prior methods satisfy only one at a time, platform-agnostic structural consistency at matched stages and developmental continuity across stages, which we establish separately and then combine. SPHERE thus turns atlas continuity from a figure drawn in presentation into a computable, stress-tested and physically realizable object.

<img width="1820" height="1496" alt="image" src="https://github.com/user-attachments/assets/d92a565b-2ccd-48e9-8b48-fe4116024025" />



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


