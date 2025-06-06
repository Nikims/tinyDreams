
# TinyDreamsI: Next Frame Prediction with Vision Transformer and CNN Decoder

This project implements a neural network model for predicting the next frame in a sequence, given the current frame and associated control inputs. It's built using PyTorch and leverages a Vision Transformer (ViT) architecture with a convolutional tokenizer and a CNN decoder head.

## What is it?

TinyDreamsI is an attempt to build a model that can learn the dynamics of a visual environment based on observed frames and the actions taken within that environment. The core idea is to encode the current visual state, combine it with information about the controls being applied, and then decode this combined representation back into a predicted future visual state (the next frame).

## Architecture Overview

The model follows a general encoder-decoder structure:

1.  **Input:** Takes the current image frame (RGB) and a vector of sensor/control inputs.
2.  **Image Tokenization:** The input image is processed to extract a sequence of visual tokens. This version uses a **Convolutional Tokenizer** based on ResNet blocks for hierarchical feature extraction before flattening into tokens.
3.  **Action Processing:** The sensor/control input vector is processed through a small MLP to produce an "action token" in the same embedding space as the image tokens.
4.  **Transformer Encoder:** The image tokens and the action token (and optionally registrar tokens) are fed into a Transformer encoder.
    *   The Transformer can use either a standard concatenated approach (action token prepended to image tokens) or an **Interleaved Cross-Attention** mechanism where image tokens attend to themselves and then cross-attend to the action token within each layer.
    *   **Registrar Tokens** can be included to potentially capture global context or act as a bottleneck.
5.  **CNN Decoder Head:** The processed tokens from the Transformer (specifically the image tokens) are reshaped back into a 2D grid and passed through a **Convolutional Decoder** based on ResNet blocks. This decoder upsamples the feature grid to reconstruct the predicted next frame.
6.  **Output:** The model outputs a predicted image frame (RGB).

## Key Features & Implementation Details

*   **Vision Transformer Backbone:** Utilizes a standard Transformer encoder structure for processing visual tokens.
*   **Convolutional Tokenizer:** Replaces the standard `PatchEmbed` with a series of downsampling convolutional layers (specifically, ResNet blocks with stride 2) to create image tokens. This allows for a more hierarchical and potentially robust initial feature extraction.
*   **CNN Decoder Head:** Uses a series of upsampling convolutional layers (ResNet blocks with nearest-neighbor upsampling) to reconstruct the image from the Transformer's output tokens.
*   **Action Token Integration:** Sensor/control inputs are embedded into a single token and integrated into the Transformer sequence.
*   **Interleaved vs. Concatenated Attention:** Supports two modes for integrating the action token within the Transformer layers.
*   **Registrar Tokens:** Optional learnable tokens can be added to the sequence.
*   **ResNet Blocks:** Both the tokenizer and decoder use custom `ResNetBlock` implementations with Instance Normalization and GELU/ReLU activations.
*   **Mixed Precision Training:** Uses `torch.cuda.amp.GradScaler` for potentially faster and more memory-efficient training on compatible hardware.
*   **Gradient Checkpointing:** Optional `torch.utils.checkpoint.checkpoint` can be used to reduce memory usage during training at the cost of computation.
*   **Loss Functions:** Primarily uses L1 loss for pixel reconstruction. Optional losses include Sobel edge loss and FFT magnitude loss.
*   **GAN Training (Optional):** Includes an optional unconditional CNN Discriminator for adversarial training. Supports soft and noisy labels for the discriminator.
*   **Checkpointing:** Saves and loads model state, optimizer state, and scaler state, with basic compatibility checks.
*   **Data Loading:** Custom `FramePredictionDataset` and `DataLoader` to handle image files and associated JSONL log data. Includes basic error handling for missing files or malformed data.

## Configuration

Most of the model's architecture and training hyperparameters are defined at the top of the `train.py` script as global constants. This includes:

*   Image dimensions and patch size
*   Transformer dimensions (embed dim, heads, layers, FFN scale)
*   Tokenizer and Decoder CNN filter sizes and ResNet block counts
*   Action MLP architecture
*   Training parameters (learning rate, batch size, epochs)
*   Loss weights
*   GAN parameters (if enabled)
*   Checkpointing behavior

**Note:** Changing certain architectural parameters (like patch size, embed dim, number of layers, filter sizes, ResNet block counts, or using/not using ConvTokenizer/Interleaved Attention/Registrar Tokens) will likely make existing checkpoints incompatible. The loading logic includes checks for some of these.

## Data Format

The model expects data in a specific format:

*   A root directory (`RECORDINGS_DIR`).
*   A JSONL file (`JSONL_LOG_FILENAME`) within that directory containing one JSON object per line.
*   Each JSON object should ideally contain:
    *   `frame_filename`: The path to the image file relative to `RECORDINGS_DIR`.
    *   `controls`: A list of floats or a dictionary (if `SENSOR_INPUT_DIM` is 4 and expects `leftStick`/`rightStick` structure) representing the control inputs *before* the frame was captured.
    *   `timestamp` or `server_timestamp_ns`: Used for sorting the log entries.

The dataset loader pairs consecutive log entries to create `(current_frame, controls, next_frame)` triplets.

## Getting Started

1.  **Prerequisites:**
    *   Python 3.7+
    *   PyTorch (with CUDA support recommended)
    *   `torchvision`
    *   `Pillow`
    *   `numpy`
2.  **Data:** Place your image files and the `recording_log.jsonl` file in the directory specified by `RECORDINGS_DIR`. Ensure the paths in the JSONL file are correct relative to this directory.
3.  **Configuration:** Adjust the constants at the top of `train.py` to match your data dimensions, desired model architecture, and training preferences.
4.  **Run Training:** Execute the script: `python train.py`

The script will automatically attempt to resume from the latest compatible checkpoint if `RESUME_FROM_LATEST` is True. Checkpoints and the final model state will be saved in the `RECORDINGS_DIR`.

## Future Work / Ideas

*   Explore different tokenizer and decoder architectures (e.g., more complex CNNs, attention-based decoders).
*   Investigate different methods for integrating action/sensor data.
*   Implement conditional GAN training.
*   Add more sophisticated loss functions (e.g., perceptual loss, SSIM).
*   Improve data loading robustness and efficiency.
*   Add evaluation metrics beyond training loss.
*   Implement a separate inference script.

---
