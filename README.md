# HolotomocuPy

## Overview

Holotomography is a coherent imaging technique that provides three-dimensional reconstruction of a sample’s complex refractive index by integrating holography principles with tomographic methods. This approach is particularly suitable for micro- and nano-tomography instruments at the latest generation of synchrotron sources.

This software package presents a family of novel algorithms, encapsulated in an efficient implementation for X-ray holotomography reconstruction. 

## Key features

* Based on Python, GPU acceleration with cuPy (GPU-accelerated numPy). 

* Regular operators (tomographic projection, Fresnel propagator, scaling, shifts, etc.) and processing methods are implemented and can be reused.

* Jupyter notebooks give examples of full pipelines for experimental data reconstruction.

* New operators/processing methods can be added by users. Implemented Python decorator @gpu_batch splits data into chunks if data do not fit into GPU memory.

* Pipeline multi-GPU data processing with CUDA streams within cuPy allows significantly reduced time for some CPU-GPU memory transfers.


## Installation

```bash
conda create -n holotomocupy -c conda-forge cupy dxchange setuptools matplotlib psutil jupyter matplotlib-scalebar
conda activate holotomocupy
cd holotomocupy; pip install -e .
```

## Performance tests

see experimental/performance_tests and Readme.txt in it


## Brain processing pipeline

see experimental/Y350a_dist1234
