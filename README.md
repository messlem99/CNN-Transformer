# CNN-Transformer for Active Cell Balancing with Proximal Policy Optimization

## Overview

This repository contains the implementation of a reinforcement learning agent for active cell balancing. The agent utilizes the Proximal Policy Optimization (PPO) algorithm with a custom feature extractor that combines a 1D Convolutional Neural Network (CNN) for local feature extraction and a Transformer for modeling global dependencies among battery cells. This architecture aims to capture both short-range correlations and long-range interactions within the battery pack to achieve effective balancing.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [CNN-Transformer Architecture](#cnn-transformer-architecture)
- [Proximal Policy Optimization](#proximal-policy-optimization)
- [License](#license)
- [Contributing](#contributing)
- [References](#references)

## Features
- **Environment:**
  ```bash 
  https://github.com/messlem99/Battery_Cell_Balancing
- **CNN-Transformer Feature Extractor:**  
  Combines a 1D Convolutional Neural Network (1D-CNN) for local feature extraction with a Transformer encoder for modeling global dependencies.
- **Custom PPO Policy:**  
  A tailored PPO policy integrates the CNN-Transformer extractor for enhanced decision making.
- **Logging and Checkpointing:**  
  Uses TensorBoard for monitoring training metrics and callback-based checkpoint saving.
- **End-to-End Training Pipeline:**  
  Supports vectorized environments and optimized hyperparameters for efficient PPO training.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/messlem99/CNN-Transformer.git
   cd CNN-Transformer
## Usage
1. **Training the Model**
To train the PPO model with the CNN-Transformer feature extractor ensure you import the environment
## CNN-Transformer Architecture
The architecture integrates two key modules:
1. **1D-CNN for Local Feature Extraction**
- Input: Historical per-cell features (voltage and SOC) arranged as a 1D sequence.
- Operation: Two convolutional layers with LeakyReLU activations, layer normalization, and dropout capture local patterns among adjacent cells.
2. **Transformer Encoder for Global Dependency Modeling:
- Reshaping: The CNN output is reshaped so that each cell acts as a token.
- Attention Mechanism: A multi-head self-attention Transformer encoder captures long-range dependencies across the battery pack.
- Global Pooling: Averages features across cells to obtain a pack-level representation.
2. **Feature Aggregation and Final Layers:**
- Concatenation: Combines the global features with derived features (including current load)
- Fully Connected Layers: Processes the combined vector to produce the final feature representation for the PPO policy.
## Proximal Policy Optimization
```bash
@misc{schulman2017,
	title={Proximal Policy Optimization Algorithms}, 
	author={John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
	year={2017},
	eprint={1707.06347},
	archivePrefix={arXiv},
	primaryClass={cs.LG},
	url={https://arxiv.org/abs/1707.06347}, 
}
```
## License
- This project is licensed under the MIT License. See the LICENSE file for details.
## Contributing
Contributions and enhancements are welcome. To get started:
- Fork the repository and create your feature branch.
- Submit pull requests for review.
## Citing
To cite this project in publications:
```bash
@misc{CNN-Transformer2025,
  author       = {Messlem Abdelkader and Messlem Youcef and Safa Ahmed},
  title        = {Hybrid Convolutional Neural Network with Transformer architecture for feature extraction within a Proximal Policy Optimization (PPO) RL framework},
  year         = {2025},
  howpublished = {\url{https://github.com/messlem99/CNN-Transformer}},
}
```
## References
- J. Li, Q. Xu, X. He, Z. Liu, D. Zhang, R. Wang, R. Qu, and G. Qiu, “Cfformer: Cross cnn-transformer channel attention and spatial feature fusion for improved segmentation of low quality medical images,” 2025. [Online]. Available: https://arxiv.org/abs/2501.03629
- 
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017. [Online]. Available: https://arxiv.org/abs/1707.06347
- J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “High-dimensional continuous control using generalized advantage estimation,” 2018. [Online]. Available: https://arxiv.org/abs/1506.02438
- A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-baselines3: Reliable reinforcement learning implementations,” Journal of Machine Learning Research, vol. 22, no. 268, pp. 1–8, 2021. [Online]. Available: http://jmlr.org/papers/v22/ 20-1364.html
- “CS231N Convolutional Neural Networks for Visual Recognition.” https://cs231n.github.io/convolutional-networks/
- “Tutorial 6: Transformers and Multi-Head Attention — UvA DL Notebooks v1.2 documentation.” https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
