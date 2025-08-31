# FastMRI: High-Quality Brain MRI Reconstruction
# Overview
This project accelerates MRI imaging using Compressed Sensing (CS) to reconstruct high-quality brain MRIs from undersampled k-space data, reducing scan times while preserving diagnostic quality. We developed DR-CAM-GAN, a U-net-based model with Dilated Residual (DR) networks and Channel Attention Mechanisms (CAM), and later adopted WGAN-GP to improve training stability. The project uses the NYU fastMRI dataset and achieves significant performance gains.
Key Features

DR-CAM-GAN: U-net architecture with DR networks and CAM for enhanced feature extraction and reduced noise.
WGAN-GP: Uses Wasserstein loss with gradient penalty for stable training at 4×, 8×, and 10× acceleration.
Dataset: NYU fastMRI brain MRI dataset.
Performance:
DR-CAM-GAN: PSNR: 29.1793 dB, SSIM: 0.7834
WGAN-GP: PSNR: 31.1499 dB, SSIM: 0.8302


Implementation: PyTorch, trained on NVIDIA RTX 3090 GPU with Adam optimizer.

# Methodology

DR-CAM-GAN: Trained with binary cross-entropy loss, leveraging CAM with global max/avg pooling to emphasize anatomical details.
WGAN-GP: Employs Wasserstein loss and gradient penalty for improved convergence and stability.
Evaluation Metrics:
Peak Signal-to-Noise Ratio (PSNR): Measures pixel-level accuracy.
Structural Similarity Index (SSIM): Assesses perceptual quality.



# Challenges

DR-CAM-GAN:
Mode collapse from Jensen-Shannon divergence, reducing sample diversity.
Unstable convergence at 8× acceleration for complex brain MRIs.


WGAN-GP Advantages:
Smoother gradients via Wasserstein loss, avoiding vanishing gradient issues.
Gradient penalty ensures stable training over weight clipping.



# Results



Model
PSNR (dB)
SSIM



DR-CAM-GAN
29.1793
0.7834


WGAN-GP
31.1499
0.8302


WGAN-GP significantly improves reconstruction quality, enhancing diagnostic potential for brain imaging.
Installation
# Clone the repository
git clone https://github.com/JohnYechanJo/FastMRI-High-Quality-Brain-MRI-Reconstruction.git
cd FastMRI-High-Quality-Brain-MRI-Reconstruction

# Install dependencies
pip install -r requirements.txt

Usage

Download the NYU fastMRI brain MRI dataset.
Update the dataset path in config.py.
Train the model:python train.py --model wgan-gp --acceleration 8


Evaluate the model:python evaluate.py --model wgan-gp --checkpoint path/to/checkpoint



Requirements

Python 3.8+
PyTorch
NVIDIA GPU (e.g., RTX 3090)
Dependencies listed in requirements.txt

Future Work

Implement StyleGAN2 or Progressive GAN for ultra-high-resolution reconstructions.
Extend to fastMRI knee/prostate datasets for multi-organ applications.
Optimize with mixed-precision training or lightweight architectures.
Develop 3D CS-MRI for volumetric brain imaging to aid surgical planning.

References

Knoll et al., Radiol Artif Intell, 2020. DOI:10.1148/ryai.2020190007
Li et al., Sensors, 2023. DOI:10.3390/s23196955
Gulrajani et al., NeurIPS, 2017. arXiv:1704.00028
Ledig et al., CVPR, 2017. DOI:10.1109/CVPR.2017.19

License
MIT License
Acknowledgments

NYU fastMRI for providing the brain MRI dataset.
John Yechan Jo for project development.
