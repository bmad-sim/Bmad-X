# Bmad-X

Experimental Bmad Code transcribed in Python with Numba and Pytorch support.

## Installation

```bash
git clone https://github.com/bmad-sim/Bmad-X.git
# pytorch-cuda on Windows or Linux.  Use environment-macos.yml for pytorch-cpu on MacOS.
conda env create -f environment.yml
conda activate bmadx
```

For a development installation of Bmad-X, run the following after creating the environment:

```bash
python -m pip install -e .
```

## Cite

```bibtex
@inproceedings{gonzalez-aguilera:ipac2023-wepa065,
 title        = {Towards fully differentiable accelerator modeling},
 author       = {Gonzalez-Aguilera, J. and Kim, Y.-K. and Roussel, R. and Edelen, A. and Mayes, C.},
 year         = 2023,
 month        = {05},
 booktitle    = {Proc. IPAC'23},
 publisher    = {JACoW Publishing, Geneva, Switzerland},
 series       = {IPAC'23 - 14th International Particle Accelerator Conference},
 number       = 14,
 pages        = {2797--2800},
 doi          = {10.18429/JACoW-IPAC2023-WEPA065},
 isbn         = {978-3-95450-231-8},
 issn         = {2673-5490},
 url          = {https://indico.jacow.org/event/41/contributions/2122},
 %  booktitle = {Proc. 14th International Particle Accelerator Conference},
 paper        = {WEPA065},
 venue        = {Venice, Italy},
 language     = {English}
}
```
