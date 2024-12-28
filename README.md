# VAE Image Generation from Scratch

This repository contains code for generating images using a Variational Autoencoder (VAE) implemented from scratch in PyTorch. The model was trained on the CelebA dataset (or your dataset). This project aims to provide a clear and understandable implementation of VAEs, making it a valuable resource for learning and experimentation.

## Overview

Variational Autoencoders are powerful generative models that learn a latent representation of data, allowing for the generation of new samples. This project provides a from-scratch implementation of a VAE using PyTorch, focusing on clarity and educational value.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Rahul7-77/Variational-AutoEncoder.git](https://github.com/Rahul7-77/Variational-AutoEncoder.git)
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd Variatonal-AutoEncoder
    ```

3.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    (Create a `requirements.txt` file using `pip freeze > requirements.txt` after installing the necessary packages: `torch`, `torchvision`, `matplotlib`, `numpy`)

5.  **Place your trained model weights:** Place your saved model weights file (e.g., `final_model_weights.pth`) in the project's root directory.

## Usage

To generate images:

```bash
python vae-generate.py --samples=10

We can vary number of samples we want
