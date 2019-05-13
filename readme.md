
This is an implementation for [On Convexity and Bounds of Fairness-aware Classification, *Yongkai Wu, Lu Zhang, Xintao Wu*](https://dl.acm.org/doi/10.1145/3308558.3313723) in WWW'19.

# Development
1. Due to the compatibility of CVXPY, this implementation works on Linux.
2. Our implementation is based on Python 3.6.
2. The python distribution [Anaconda](https://www.anaconda.com) or [Miniconda](https://repo.continuum.io/miniconda/) is highly recommended. Since we utilize the environment management tool `conda`, Miniconda is minimal and sufficient.


# Reproduction
To re-produce this repository:
1.  Recover the environment by `conda env create --file conda-env.txt --name YOUR_ENV_NAME`.
2.  Go into the new environment and install `dccp` by `pip install dccp==0.1.6`.
3.  run `python main.py`.

## BibTex

```
@inproceedings{10.1145/3308558.3313723,
author = {Wu, Yongkai and Zhang, Lu and Wu, Xintao},
title = {On Convexity and Bounds of Fairness-Aware Classification},
year = {2019},
isbn = {9781450366748},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3308558.3313723},
doi = {10.1145/3308558.3313723},
booktitle = {The World Wide Web Conference},
pages = {3356–3362},
numpages = {7},
keywords = {algorithmic bias, classification;constrained optimization, Fairness-aware machine learning},
location = {San Francisco, CA, USA},
series = {WWW '19}
}
```