# PassREfinder-FL
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of [*PassREfinder-FL: Privacy-Preserving Credential Stuffing Risk Prediction via Graph-Based Federated Learning for Representing Password Reuse between Websites*](https://www.sciencedirect.com/science/article/pii/S0957417425036954), accepted at Elsevier Expert Systems with Applications (ESWA) 2025.

## System Requirements
- **GPU**: 1x NVIDIA RTX 3090 24G
- **CUDA**: 12.1

### Installation

1. **Install Anaconda**
   
   Download and install [Anaconda](https://www.anaconda.com/download).

2. **Clone the Repository**
   ```bash
   git clone https://github.com/jaehanwork/PassREfinder-FL.git
   cd PassREfinder-FL
   ```

3. **Set Up Environment**
   ```bash
   conda env create -f environment.yml -n passrefinder-fl
   conda activate passrefinder-fl
   pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
   ```

### Decompress 
```
tar -zxvf data/data.tar.gz -C data
```

### Run

```bash
./run.sh
```


## Citation

```bibtex
@article{kim2025passrefinder,
  title={PassREfinder-FL: Privacy-Preserving Credential Stuffing Risk Prediction via Graph-Based Federated Learning for Representing Password Reuse between Websites},
  author={Kim, Jaehan and Song, Minkyoo and Seo, Minjae and Jin, Youngjin and Shin, Seungwon and Kim, Jinwoo},
  journal={Expert Systems with Applications},
  pages={130111},
  year={2025},
  publisher={Elsevier}
}
```