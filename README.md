<div align="center">
<img src="./FastMRI-Brain-MRI-Reconstruction.png" alt="FastMRI Brain MRI Reconstruction">
<br>
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=4&height=80">
<h1>FastMRI: High-Quality Brain MRI Reconstruction</h1>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction)](https://github.com/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction)](https://github.com/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction/issues)
<br>

## Contributors
[![GitHub Contributors](https://img.shields.io/github/contributors-anon/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction)](https://github.com/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction/graphs/contributors)
<table>
  <tr>
    <td align="center"><a href="https://github.com/JohnYechanJo"><img src="https://avatars.githubusercontent.com/u/131790222?v=4" width="100px;" alt=""/><br /><sub><b>John Yechan Jo</b></sub></a><br /></td>
  </tr>
</table>
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=4&height=80&section=footer&fontSize=80">
</div>

## Overview
This project accelerates MRI imaging using Compressed Sensing (CS) to reconstruct high-quality brain MRIs from undersampled k-space data, reducing scan times while preserving diagnostic quality. We developed **DR-CAM-GAN**, a U-net-based model with Dilated Residual (DR) networks and Channel Attention Mechanisms (CAM), and later adopted **WGAN-GP** to improve training stability, achieving superior performance on the NYU fastMRI dataset.

## Key Features
- **DR-CAM-GAN**: U-net architecture with Arthurs and CAM for enhanced feature extraction and reduced noise.
- **WGAN-GP**: Uses Wasserstein loss with gradient penalty for stable training at 4×, 8×, and 10× acceleration.
- **Dataset**: NYU fastMRI brain MRI dataset.
- **Performance**:
  - DR-CAM-GAN: PSNR: 29.1793 dB, SSIM: 0.7834
  - WGAN-GP: PSNR: 31.1499 dB, SSIM: 0.8302
- **Implementation**: PyTorch, trained on NVIDIA RTX 3090 GPU with Adam optimizer.

## Methodology
- **DR-CAM-GAN**: Trained with binary cross-entropy loss, leveraging CAM with global max/avg pooling to emphasize anatomical details.
- **WGAN-GP**: Uses Wasserstein loss with gradient penalty for improved convergence and stability.
- **Evaluation Metrics**:
  - Peak Signal-to-Noise Ratio (PSNR): Measures pixel-level accuracy.
  - Structural Similarity Index (SSIM): Assesses perceptual quality.

## Challenges
- **DR-CAM-GAN**:
  - Mode collapse due to Jensen-Shannon divergence, reducing sample diversity.
  - Unstable convergence at 8× acceleration for complex brain MRIs.
- **WGAN-GP Advantages**:
  - Smoother gradients via Wasserstein loss, avoiding vanishing gradient issues.
  - Gradient penalty ensures stable training over weight clipping.

## Results
| Model         | PSNR (dB) | SSIM  |
|---------------|-----------|-------|
| DR-CAM-GAN    | 29.1793   | 0.7834 |
| WGAN-GP       | 31.1499   | 0.8302 |

WGAN-GP significantly improves reconstruction quality, enhancing diagnostic potential for brain imaging.

## Installation
```bash
# Clone the repository
git clone https://github.com/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction.git
cd FastMRI-High-Quality-Brain-MRI-Reconstruction

# Install dependencies
pip install -r requirements.txt
