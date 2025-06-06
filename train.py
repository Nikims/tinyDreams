import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
import logging
import time
import math
import re
import random
from torch.cuda.amp import GradScaler
from torch.utils.checkpoint import checkpoint

# --- Configuration ---
RECORDINGS_DIR = "recordings_n"
JSONL_LOG_FILENAME = "recording_log.jsonl"

INPUT_IMG_HEIGHT_GLOBAL = 64*4
INPUT_IMG_WIDTH_GLOBAL = 64*4
IMG_CHANNELS = 3
SENSOR_INPUT_DIM = 4 # e.g., [left_stick_x, left_stick_y, right_stick_x, right_stick_y]

# --- ViT Backbone Configuration ---
PATCH_SIZE = 32
EMBED_DIM = 280+28
NUM_REGISTRAR_TOKENS = 12

# --- Convolutional Tokenizer Configuration (if USE_CONV_TOKENIZER is True) ---
USE_CONV_TOKENIZER = True
TOKENIZER_CONV_OUT_CHANNELS = [64, 128, 256, 256, EMBED_DIM]
TOKENIZER_CONV_KERNEL_SIZE = 3 # Original kernel size, ResNetBlocks will primarily use 3x3
TOKENIZER_NUM_RESBLOCKS_PER_STAGE = 2 # Number of ResNetBlocks per downsampling stage in Tokenizer

# --- CNN Decoder Configuration ---
DECODER_CNN_FILTER_SIZES = [240, 180, 96, 64, 48] # Length 5 for PATCH_SIZE=32
DECODER_NUM_RESBLOCKS_PER_STAGE = 2 # Number of ResNetBlocks per upsampling stage in Decoder


# --- Transformer Configuration ---
TRANSFORMER_NHEAD = 14
TRANSFORMER_NUM_ENCODER_LAYERS = 14
TRANSFORMER_DIM_FEEDFORWARD_SCALE = 3
TRANSFORMER_DROPOUT = 0.1
USE_INTERLEAVED_ATTENTION = False

MODEL_SAVE_NAME_BASE = "nextframe_predictor_vit_action_token_cnndec"
MODEL_SAVE_NAME = f"{MODEL_SAVE_NAME_BASE}"
if USE_CONV_TOKENIZER:
    MODEL_SAVE_NAME += f"_convtok_res{TOKENIZER_NUM_RESBLOCKS_PER_STAGE}" # Indicate ResNet Tokenizer
if not USE_INTERLEAVED_ATTENTION:
    MODEL_SAVE_NAME += "_concat"
if NUM_REGISTRAR_TOKENS > 0:
    MODEL_SAVE_NAME += f"_reg{NUM_REGISTRAR_TOKENS}"
MODEL_SAVE_NAME += f"_decres{DECODER_NUM_RESBLOCKS_PER_STAGE}"


RESUME_FROM_LATEST = True
MODEL_ARCHITECTURE_TYPE = "vit_transformer_cnndec_resnet" # Updated arch type

# --- Action Token Configuration ---
ACTION_PROCESS_MLP_HIDDEN_LAYERS = [15,31,63]

# --- Training Hyperparameters ---
LEARNING_RATE = 6e-4
BATCH_SIZE = 12*3
NUM_EPOCHS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Loss Weights ---
RECONSTRUCTION_LOSS_WEIGHT = 0.4
SOBEL_WEIGHT = 0.0
FFT_MAG_LOSS_WEIGHT = 0
TRANSFORMER_BOTTLENECK_NOISE_STD_FACTOR = 0.00

# --- GAN Configuration (Unconditional Discriminator) ---
USE_GAN_TRAINING = False
DISCRIMINATOR_FILTERS = [32, 64, 128]
DISCRIMINATOR_MLP_ARCHITECTURE = [512, 256, 1]
DISCRIMINATOR_LR = 1e-4
ADVERSARIAL_LOSS_WEIGHT = 0.01
D_TRAIN_RATIO = 1

USE_SOFT_LABELS = True
USE_NOISY_LABELS = True
D_REAL_LABEL_LOW = 0.0
D_REAL_LABEL_HIGH = 0.2
D_FAKE_LABEL_LOW = 0.8
D_FAKE_LABEL_HIGH = 1.0
G_TARGET_LABEL_LOW = 0.0
G_TARGET_LABEL_HIGH = 0.2
LABEL_NOISE_STRENGTH = 0.05

if USE_GAN_TRAINING:
    MODEL_SAVE_NAME += "_gan_uncondD"
    if USE_SOFT_LABELS and USE_NOISY_LABELS:
        MODEL_SAVE_NAME += "_soft_noisy_inv_labels"
    elif USE_SOFT_LABELS:
        MODEL_SAVE_NAME += "_soft_inv_labels"
    elif USE_NOISY_LABELS:
        MODEL_SAVE_NAME += "_noisy_inv_labels"
    else:
        MODEL_SAVE_NAME += "_inv_labels"
    MODEL_SAVE_NAME += f"_dratio{D_TRAIN_RATIO}"

USE_GRADIENT_CHECKPOINTING = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Configuration Validation ---
if EMBED_DIM % TRANSFORMER_NHEAD != 0:
    raise ValueError("EMBED_DIM must be divisible by TRANSFORMER_NHEAD.")

if PATCH_SIZE <= 0 or (PATCH_SIZE > 1 and (PATCH_SIZE & (PATCH_SIZE - 1)) != 0):
    raise ValueError("PATCH_SIZE must be a power of 2 and >= 2 (or 1 if not using ConvTokenizer).")

if USE_CONV_TOKENIZER:
    if not isinstance(TOKENIZER_CONV_OUT_CHANNELS, list) or not TOKENIZER_CONV_OUT_CHANNELS:
        raise ValueError("TOKENIZER_CONV_OUT_CHANNELS must be a non-empty list if USE_CONV_TOKENIZER is True.")
    if PATCH_SIZE < 2 : # ConvTokenizer implies PATCH_SIZE must be a power of 2 >= 2.
        raise ValueError("ConvTokenizer requires PATCH_SIZE to be a power of 2, >= 2.")

    expected_num_stages_tok = int(math.log2(PATCH_SIZE))
    if len(TOKENIZER_CONV_OUT_CHANNELS) != expected_num_stages_tok:
        raise ValueError(
            f"Length of TOKENIZER_CONV_OUT_CHANNELS ({len(TOKENIZER_CONV_OUT_CHANNELS)}) "
            f"must be log2(PATCH_SIZE) ({expected_num_stages_tok}) when USE_CONV_TOKENIZER is True."
        )
    if TOKENIZER_CONV_OUT_CHANNELS[-1] != EMBED_DIM:
        raise ValueError(
            f"The last channel in TOKENIZER_CONV_OUT_CHANNELS ({TOKENIZER_CONV_OUT_CHANNELS[-1]}) "
            f"must be equal to EMBED_DIM ({EMBED_DIM})."
        )
    if TOKENIZER_NUM_RESBLOCKS_PER_STAGE < 1:
        raise ValueError("TOKENIZER_NUM_RESBLOCKS_PER_STAGE must be at least 1.")

elif not USE_CONV_TOKENIZER and PATCH_SIZE <= 0: # For original PatchEmbed
    raise ValueError("PATCH_SIZE must be positive for PatchEmbed.")

# CNN Decoder Validation
num_upsample_layers_dec = int(math.log2(PATCH_SIZE))
if len(DECODER_CNN_FILTER_SIZES) != num_upsample_layers_dec:
    raise ValueError(
        f"CNNDecoderHead: Length of DECODER_CNN_FILTER_SIZES ({len(DECODER_CNN_FILTER_SIZES)}) "
        f"must be num_upsample_layers (log2(PATCH_SIZE) = {num_upsample_layers_dec})."
    )
if DECODER_NUM_RESBLOCKS_PER_STAGE < 1:
    raise ValueError("DECODER_NUM_RESBLOCKS_PER_STAGE must be at least 1.")


def get_padding(kernel_size, stride, dilation):
    if stride != 1:
         logging.warning(f"get_padding called with stride {stride}. Formula assumes stride 1 for 'same' padding.")
    return dilation * (kernel_size - 1) // 2

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation_type="gelu", is_output_layer=False):
        super().__init__()
        self.is_output_layer = is_output_layer

        if activation_type == "gelu":
            self.activation_fn = nn.GELU
        elif activation_type == "relu":
            self.activation_fn = lambda: nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act1 = self.activation_fn()

        # Second convolution: bias is True only if it's the final output layer of the tokenizer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=is_output_layer)

        if not is_output_layer:
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act2 = self.activation_fn() # Activation after addition

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Shortcut convolution always has bias=False as it's typically followed by a norm
            shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            shortcut_norm = nn.InstanceNorm2d(out_channels, affine=True)
            self.shortcut = nn.Sequential(shortcut_conv, shortcut_norm)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)

        if not self.is_output_layer:
            out = self.norm2(out)

        out += identity

        if not self.is_output_layer:
            out = self.act2(out)
        return out

class InterleavedCrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.norm1_img = nn.LayerNorm(d_model)
        self.self_attn_img = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1_img = nn.Dropout(dropout)

        self.norm2_img_query = nn.LayerNorm(d_model)
        self.norm2_action_kv = nn.LayerNorm(d_model)
        self.cross_attn_img_to_action = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2_img = nn.Dropout(dropout)

        self.norm3_img = nn.LayerNorm(d_model)
        self.ffn_img = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout3_img = nn.Dropout(dropout)

    def forward(self, img_src, action_context):
        img_src_normed_sa = self.norm1_img(img_src)
        img_attn_out, _ = self.self_attn_img(img_src_normed_sa, img_src_normed_sa, img_src_normed_sa)
        img_src = img_src + self.dropout1_img(img_attn_out)

        img_src_normed_ca_query = self.norm2_img_query(img_src)
        action_context_normed_ca_kv = self.norm2_action_kv(action_context)
        img_cross_attn_out, _ = self.cross_attn_img_to_action(
            query=img_src_normed_ca_query,
            key=action_context_normed_ca_kv,
            value=action_context_normed_ca_kv
        )
        img_src = img_src + self.dropout2_img(img_cross_attn_out)

        img_src_normed_ffn = self.norm3_img(img_src)
        img_ffn_out = self.ffn_img(img_src_normed_ffn)
        img_src = img_src + self.dropout3_img(img_ffn_out)

        return img_src

class StandardTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src_normed = self.norm1(src)
        attn_output, _ = self.self_attn(src_normed, src_normed, src_normed)
        src = src + self.dropout1(attn_output)

        src_normed = self.norm2(src)
        ffn_output = self.ffn(src_normed)
        src = src + self.dropout2(ffn_output)
        return src

class FramePredictionDataset(Dataset):
    def __init__(self, recordings_dir, log_filename, img_transform=None,
                 dataloader_img_height=INPUT_IMG_HEIGHT_GLOBAL, dataloader_img_width=INPUT_IMG_WIDTH_GLOBAL,
                 expected_controls_len=SENSOR_INPUT_DIM):
        self.recordings_dir = recordings_dir
        self.log_path = os.path.join(recordings_dir, log_filename)
        self.img_transform = img_transform
        self.dataloader_img_height = dataloader_img_height
        self.dataloader_img_width = dataloader_img_width
        self.expected_controls_len = expected_controls_len
        self.sequences = self._load_sequences()
        if not self.sequences: logging.warning(f"No valid sequences found in {self.log_path}.")
        logging.info(f"Loaded {len(self.sequences)} sequences. DataLoader target size: {dataloader_img_height}x{dataloader_img_width}. Expected controls length: {self.expected_controls_len}")

    def _load_sequences(self):
        sequences = []
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                log_data = [json.loads(line) for line in f]
        except Exception as e:
            logging.error(f"Error reading log file {self.log_path}: {e}")
            return []
        log_data.sort(key=lambda x: x.get('timestamp', x.get('server_timestamp_ns', 0)))
        skipped_count = 0
        for i in range(len(log_data) - 1):
            current_entry = log_data[i]
            next_entry = log_data[i+1]
            current_img_fn = current_entry.get("frame_filename", "")
            next_img_fn = next_entry.get("frame_filename", "")
            current_img_path = os.path.join(self.recordings_dir, current_img_fn)
            next_img_path = os.path.join(self.recordings_dir, next_img_fn)

            controls_data = current_entry.get("controls", {})
            if isinstance(controls_data, list) and len(controls_data) >= self.expected_controls_len:
                 raw_controls = controls_data[:self.expected_controls_len]
            elif isinstance(controls_data, dict) and self.expected_controls_len == 4:
                try:
                    raw_controls = [
                        float(controls_data.get('leftStick', {}).get('x', 0.0)),
                        float(controls_data.get('leftStick', {}).get('y', 0.0)),
                        float(controls_data.get('rightStick', {}).get('x', 0.0)),
                        float(controls_data.get('rightStick', {}).get('y', 0.0))
                    ]
                except (ValueError, TypeError):
                    skipped_count += 1
                    continue
            else:
                if self.expected_controls_len == 0:
                    raw_controls = []
                elif isinstance(controls_data, list) and len(controls_data) < self.expected_controls_len:
                    raw_controls = controls_data + [0.0] * (self.expected_controls_len - len(controls_data))
                else:
                    skipped_count += 1
                    continue

            if not all([current_img_fn, next_img_fn,
                        os.path.exists(current_img_path), os.path.exists(next_img_path)]):
                skipped_count += 1
                continue
            try:
                if self.expected_controls_len > 0:
                    controls_tensor = torch.tensor([float(c) for c in raw_controls], dtype=torch.float32)
                    if controls_tensor.shape[0] != self.expected_controls_len:
                        skipped_count += 1
                        continue
                else:
                    controls_tensor = torch.empty(0, dtype=torch.float32)

            except Exception as e:
                skipped_count += 1
                continue
            sequences.append({
                "current_img_path": current_img_path,
                "next_img_path": next_img_path,
                "controls": controls_tensor
            })
        if skipped_count > 0:
            logging.warning(f"Skipped {skipped_count} sequences due to missing files, data, or incorrect controls format/length.")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        current_img_path = sequence["current_img_path"]
        next_img_path = sequence["next_img_path"]
        controls = sequence["controls"]

        dummy_img_tensor = torch.zeros((IMG_CHANNELS, self.dataloader_img_height, self.dataloader_img_width), dtype=torch.float32)
        if self.expected_controls_len > 0:
            dummy_controls_tensor = torch.zeros(self.expected_controls_len, dtype=torch.float32)
        else:
            dummy_controls_tensor = torch.empty(0, dtype=torch.float32)

        if self.img_transform:
            try:
                dummy_pil = Image.new('RGB', (self.dataloader_img_width, self.dataloader_img_height))
                dummy_img_tensor_transformed = self.img_transform(dummy_pil)
            except Exception:
                 dummy_img_tensor_transformed = dummy_img_tensor
        else:
            resize_transform = transforms.Compose([
                transforms.Resize((self.dataloader_img_height, self.dataloader_img_width)),
                transforms.ToTensor()
            ])
            dummy_pil = Image.new('RGB', (self.dataloader_img_width, self.dataloader_img_height))
            dummy_img_tensor_transformed = resize_transform(dummy_pil)

        try:
            current_img_pil = Image.open(current_img_path).convert('RGB')
            next_img_pil = Image.open(next_img_path).convert('RGB')
            if self.img_transform:
                current_img_for_model_input = self.img_transform(current_img_pil)
                next_img_for_model_input = self.img_transform(next_img_pil)
            else:
                 resize_transform = transforms.Compose([
                     transforms.Resize((self.dataloader_img_height, self.dataloader_img_width)),
                     transforms.ToTensor()
                 ])
                 current_img_for_model_input = resize_transform(current_img_pil)
                 next_img_for_model_input = resize_transform(next_img_pil)
        except Exception as e:
             logging.error(f"Error loading/transforming image pair {idx} ('{current_img_path}', '{next_img_path}'): {e}")
             return f"ERR_{idx}_curr_path", dummy_img_tensor_transformed, dummy_controls_tensor, f"ERR_{idx}_next_path", dummy_img_tensor_transformed

        expected_model_input_shape = (IMG_CHANNELS, self.dataloader_img_height, self.dataloader_img_width)
        if current_img_for_model_input.shape != expected_model_input_shape or \
           next_img_for_model_input.shape != expected_model_input_shape:
             logging.warning(f"Unexpected model input image shape for pair {idx}. Got {current_img_for_model_input.shape}, {next_img_for_model_input.shape}. Expected {expected_model_input_shape}. Returning dummy.")
             return f"ERR_{idx}_curr_path_shape", dummy_img_tensor_transformed, dummy_controls_tensor, f"ERR_{idx}_next_path_shape", dummy_img_tensor_transformed

        if self.expected_controls_len > 0 and controls.shape[0] != self.expected_controls_len:
            logging.warning(f"Controls length mismatch for item {idx}. Expected {self.expected_controls_len}, got {controls.shape[0]}. Returning dummy controls.")
            return current_img_path, current_img_for_model_input, dummy_controls_tensor, next_img_path, next_img_for_model_input

        return current_img_path, current_img_for_model_input, controls, next_img_path, next_img_for_model_input


class PatchEmbed(nn.Module):
    def __init__(self, img_size_h, img_size_w, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        if img_size_h % patch_size != 0 or img_size_w % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size_h or W != self.img_size_w:
            logging.warning(f"PatchEmbed input image size ({H}x{W}) doesn't match model's expected size ({self.img_size_h}x{self.img_size_w}). Attempting resize.")
            x = F.interpolate(x, size=(self.img_size_h, self.img_size_w), mode='bilinear', align_corners=False)

        x = self.proj(x)
        return x

class ConvTokenizer(nn.Module):
    def __init__(self, img_size_h, img_size_w, patch_size, in_chans, embed_dim,
                 tokenizer_conv_out_channels, tokenizer_num_resblocks_per_stage, kernel_size=3): # kernel_size is less direct now
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        if patch_size < 2 or (patch_size & (patch_size - 1)) != 0:
             raise ValueError("ConvTokenizer: patch_size must be a power of 2 and >= 2.")
        num_conv_stages = int(math.log2(patch_size))

        if len(tokenizer_conv_out_channels) != num_conv_stages:
            raise ValueError(
                f"ConvTokenizer: Length of tokenizer_conv_out_channels ({len(tokenizer_conv_out_channels)}) "
                f"must be log2(patch_size) ({num_conv_stages})."
            )
        if tokenizer_conv_out_channels[-1] != embed_dim:
            raise ValueError(
                f"ConvTokenizer: The last channel in tokenizer_conv_out_channels ({tokenizer_conv_out_channels[-1]}) "
                f"must be equal to embed_dim ({embed_dim})."
            )
        if tokenizer_num_resblocks_per_stage < 1:
            raise ValueError("ConvTokenizer: tokenizer_num_resblocks_per_stage must be at least 1.")

        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        if img_size_h % patch_size != 0 or img_size_w % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size for ConvTokenizer.")

        layers = []
        current_in_channels = in_chans

        for i in range(num_conv_stages):
            stage_out_channels = tokenizer_conv_out_channels[i]

            # First ResNetBlock in the stage handles downsampling (stride=2)
            is_first_block_the_very_last_overall = (i == num_conv_stages - 1) and \
                                                   (0 == tokenizer_num_resblocks_per_stage - 1)
            layers.append(
                ResNetBlock(
                    in_channels=current_in_channels,
                    out_channels=stage_out_channels,
                    stride=2,
                    activation_type="gelu",
                    is_output_layer=is_first_block_the_very_last_overall # CORRECTED HERE
                )
            )
            # Input channels for subsequent blocks in this stage, or for the next stage's first block
            current_block_in_channels = stage_out_channels

            # Additional ResNetBlocks at this resolution (stride=1)
            for j in range(1, tokenizer_num_resblocks_per_stage):
                is_this_block_the_very_last_overall = (i == num_conv_stages - 1) and \
                                                      (j == tokenizer_num_resblocks_per_stage - 1)
                layers.append(
                    ResNetBlock(
                        in_channels=current_block_in_channels,
                        out_channels=stage_out_channels,
                        stride=1,
                        activation_type="gelu",
                        is_output_layer=is_this_block_the_very_last_overall
                    )
                )
                # current_block_in_channels remains stage_out_channels for these blocks

            current_in_channels = stage_out_channels # Update for the next stage

        self.tokenizer = nn.Sequential(*layers)
        logging.info(f"ConvTokenizer: {num_conv_stages} stages, {tokenizer_num_resblocks_per_stage} ResNetBlock(s) per stage. "
                     f"Channels: {in_chans} -> {' -> '.join(map(str, tokenizer_conv_out_channels))}. Activation: GELU.")

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size_h or W != self.img_size_w:
            logging.warning(
                f"ConvTokenizer input image size ({H}x{W}) doesn't match model's expected size "
                f"({self.img_size_h}x{self.img_size_w}). Attempting resize."
            )
            x = F.interpolate(x, size=(self.img_size_h, self.img_size_w), mode='bilinear', align_corners=False)

        x = self.tokenizer(x)
        return x


class CNNDecoderHead(nn.Module):
    def __init__(self, embed_dim, out_chans, patch_size, num_patches_h, num_patches_w,
                 decoder_cnn_filter_sizes, decoder_num_resblocks_per_stage):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.final_img_h = num_patches_h * patch_size
        self.final_img_w = num_patches_w * patch_size

        num_upsample_layers = int(math.log2(patch_size))
        if 2**num_upsample_layers != patch_size:
            raise ValueError(f"Patch size ({patch_size}) must be a power of 2 for this CNN decoder structure.")

        if len(decoder_cnn_filter_sizes) != num_upsample_layers:
            raise ValueError(
                f"CNNDecoderHead: Length of decoder_cnn_filter_sizes ({len(decoder_cnn_filter_sizes)}) "
                f"must be num_upsample_layers ({num_upsample_layers}). Patch size {patch_size}."
            )
        if decoder_num_resblocks_per_stage < 1:
            raise ValueError("CNNDecoderHead: decoder_num_resblocks_per_stage must be at least 1.")

        layers = []
        current_channels = embed_dim # Input to the first upsampling stage's ResBlocks

        for i in range(num_upsample_layers):
            stage_resblock_out_channels = decoder_cnn_filter_sizes[i]

            layers.append(nn.Upsample(scale_factor=2, mode='nearest')) # Or 'bilinear'

            input_to_resblock_stage = current_channels
            for j in range(decoder_num_resblocks_per_stage):
                layers.append(
                    ResNetBlock(
                        in_channels=input_to_resblock_stage,
                        out_channels=stage_resblock_out_channels,
                        stride=1, # ResBlocks operate at the upsampled resolution
                        activation_type="relu", # Decoder uses ReLU for intermediate layers
                        is_output_layer=False # All ResBlocks in decoder are intermediate
                    )
                )
                input_to_resblock_stage = stage_resblock_out_channels # For subsequent ResBlocks in this stage

            current_channels = stage_resblock_out_channels # Output of this stage becomes input for next

        layers.append(nn.Conv2d(current_channels, out_chans, kernel_size=3, padding=1, bias=True))
        layers.append(nn.Sigmoid())

        self.decoder_cnn = nn.Sequential(*layers)
        logging.info(f"CNNDecoderHead: {num_upsample_layers} upsampling stages, {decoder_num_resblocks_per_stage} ResNetBlock(s) per stage. "
                     f"ResBlock channels: {embed_dim} -> {' -> '.join(map(str, decoder_cnn_filter_sizes))} -> FinalConv -> {out_chans}. Activation: ReLU.")

    def forward(self, x):
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = x.view(B, self.embed_dim, self.num_patches_h, self.num_patches_w)
        x = self.decoder_cnn(x)
        if x.shape[2] != self.final_img_h or x.shape[3] != self.final_img_w:
            logging.warning(f"CNNDecoderHead output shape {x.shape[2]}x{x.shape[3]} (expected {self.final_img_h}x{self.final_img_w}). Resizing.")
            x = F.interpolate(x, size=(self.final_img_h, self.final_img_w), mode='bilinear', align_corners=False)
        return x


class TransformerNextFrameGenerator(nn.Module):
    def __init__(self, img_channels, sensor_dim, embed_dim, patch_size,
                 nhead, num_encoder_layers, dim_feedforward_scale, dropout,
                 input_img_height, input_img_width,
                 action_process_mlp_hidden_layers,
                 decoder_cnn_filter_sizes, # Passed to CNNDecoderHead
                 transformer_bottleneck_noise_std_factor=0.0,
                 use_interleaved_attention=True,
                 num_registrar_tokens=0,
                 use_conv_tokenizer=False,
                 tokenizer_conv_out_channels=None,
                 tokenizer_conv_kernel_size=3, # Less direct impact with ResNetBlocks
                 tokenizer_num_resblocks_per_stage=1, # New
                 decoder_num_resblocks_per_stage=1    # New
                 ):
        super().__init__()
        self.img_channels = img_channels
        self.sensor_dim = sensor_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.input_img_height = input_img_height
        self.input_img_width = input_img_width
        self.transformer_bottleneck_noise_std_factor = transformer_bottleneck_noise_std_factor
        self.use_interleaved_attention = use_interleaved_attention
        self.num_registrar_tokens = num_registrar_tokens

        self.use_conv_tokenizer = use_conv_tokenizer
        self.tokenizer_conv_out_channels = tokenizer_conv_out_channels
        self.tokenizer_conv_kernel_size = tokenizer_conv_kernel_size
        self.tokenizer_num_resblocks_per_stage = tokenizer_num_resblocks_per_stage # Store

        self.num_patches_h = input_img_height // patch_size
        self.num_patches_w = input_img_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        dim_feedforward = embed_dim * dim_feedforward_scale

        logging.info(f"ViT Generator: Image Patches: {self.num_patches_h}x{self.num_patches_w} ({self.num_patches} total). Embed Dim: {embed_dim}")
        logging.info(f"Transformer: Layers: {num_encoder_layers}, Heads: {nhead}, FFN Dim: {dim_feedforward}, Dropout: {dropout}")
        logging.info(f"Using Interleaved Attention: {self.use_interleaved_attention}")

        if self.use_conv_tokenizer:
            self.patch_embed = ConvTokenizer(
                img_size_h=input_img_height, img_size_w=input_img_width,
                patch_size=patch_size, in_chans=img_channels, embed_dim=embed_dim,
                tokenizer_conv_out_channels=self.tokenizer_conv_out_channels,
                tokenizer_num_resblocks_per_stage=self.tokenizer_num_resblocks_per_stage, # Pass new param
                kernel_size=self.tokenizer_conv_kernel_size
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size_h=input_img_height, img_size_w=input_img_width,
                patch_size=patch_size, in_chans=img_channels, embed_dim=embed_dim
            )
            logging.info("Using original PatchEmbed (single Conv2d for tokenization).")

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.num_patches_h, self.num_patches_w))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        if self.num_registrar_tokens > 0:
            self.registrar_tokens = nn.Parameter(torch.zeros(1, self.num_registrar_tokens, embed_dim))
            nn.init.trunc_normal_(self.registrar_tokens, std=.02)
            logging.info(f"Using {self.num_registrar_tokens} registrar token(s).")
        else:
            self.registrar_tokens = None

        mlp_layers = []
        mlp_input_dim = sensor_dim if sensor_dim > 0 else 1
        current_dim = mlp_input_dim

        for hidden_dim in action_process_mlp_hidden_layers:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim = hidden_dim
        self.action_mlp = nn.Sequential(*mlp_layers)
        self.action_to_token_projection = nn.Linear(current_dim, embed_dim)
        self.action_token_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.action_token_pos_embed, std=.02)
        logging.info(f"Action Processing: MLP (input {mlp_input_dim} -> ... -> {current_dim}) -> Project to {embed_dim} dim token.")

        if self.use_interleaved_attention:
            self.transformer_layers = nn.ModuleList(
                [InterleavedCrossAttentionEncoderLayer(embed_dim, nhead, dim_feedforward, dropout)
                 for _ in range(num_encoder_layers)]
            )
        else:
            self.transformer_layers = nn.ModuleList(
                [StandardTransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout)
                 for _ in range(num_encoder_layers)]
            )

        self.decoder_head = CNNDecoderHead(
            embed_dim=embed_dim,
            out_chans=img_channels,
            patch_size=patch_size,
            num_patches_h=self.num_patches_h,
            num_patches_w=self.num_patches_w,
            decoder_cnn_filter_sizes=decoder_cnn_filter_sizes,
            decoder_num_resblocks_per_stage=decoder_num_resblocks_per_stage # Pass new param
        )


    def forward(self, img, actions_vec):
        B = img.shape[0]

        image_patch_features_2d = self.patch_embed(img)
        image_patch_features_with_pe_2d = image_patch_features_2d + self.pos_embed
        image_tokens_flat = image_patch_features_with_pe_2d.flatten(2)
        image_tokens_for_transformer = image_tokens_flat.transpose(1, 2)

        if self.sensor_dim == 0:
            mlp_input = torch.zeros(B, 1, device=img.device, dtype=torch.float32)
        else:
            mlp_input = actions_vec
        mlp_out = self.action_mlp(mlp_input)
        action_token = self.action_to_token_projection(mlp_out)
        action_token = action_token.unsqueeze(1)
        action_token_with_pe = action_token + self.action_token_pos_embed.expand(B, -1, -1)

        registrar_tokens_batch = None
        if self.num_registrar_tokens > 0 and self.registrar_tokens is not None:
            registrar_tokens_batch = self.registrar_tokens.expand(B, -1, -1)

        final_tokens_for_decoder = None

        if self.use_interleaved_attention:
            current_sequence_for_self_attn = image_tokens_for_transformer
            if registrar_tokens_batch is not None:
                current_sequence_for_self_attn = torch.cat([registrar_tokens_batch, current_sequence_for_self_attn], dim=1)

            processed_sequence = current_sequence_for_self_attn
            for layer in self.transformer_layers:
                if self.training and USE_GRADIENT_CHECKPOINTING:
                     def create_custom_forward(module_fn):
                         def custom_forward(*inputs):
                             return module_fn(inputs[0], inputs[1])
                         return custom_forward
                     processed_sequence = checkpoint(create_custom_forward(layer), processed_sequence, action_token_with_pe, use_reentrant=False)
                else:
                    processed_sequence = layer(processed_sequence, action_token_with_pe)

            if registrar_tokens_batch is not None:
                final_tokens_for_decoder = processed_sequence[:, self.num_registrar_tokens:, :]
            else:
                final_tokens_for_decoder = processed_sequence
        else:
            tokens_to_combine = []
            if registrar_tokens_batch is not None:
                tokens_to_combine.append(registrar_tokens_batch)
            tokens_to_combine.append(action_token_with_pe)
            tokens_to_combine.append(image_tokens_for_transformer)

            combined_sequence = torch.cat(tokens_to_combine, dim=1)
            processed_sequence = combined_sequence
            for layer in self.transformer_layers:
                if self.training and USE_GRADIENT_CHECKPOINTING:
                     processed_sequence = checkpoint(layer, processed_sequence, use_reentrant=False)
                else:
                    processed_sequence = layer(processed_sequence)

            num_prefix_tokens = 1
            if registrar_tokens_batch is not None:
                num_prefix_tokens += self.num_registrar_tokens
            final_tokens_for_decoder = processed_sequence[:, num_prefix_tokens:, :]


        if self.transformer_bottleneck_noise_std_factor > 0 and self.training:
            feature_std = final_tokens_for_decoder.std(dim=(0, 1), keepdim=True) + 1e-6
            noise = torch.randn_like(final_tokens_for_decoder) * feature_std * self.transformer_bottleneck_noise_std_factor
            final_tokens_for_decoder = final_tokens_for_decoder + noise

        predicted_output = self.decoder_head(final_tokens_for_decoder)
        return predicted_output

    def calculate_natural_output_resolution(self, input_h, input_w):
        num_patches_h = input_h // self.patch_size
        num_patches_w = input_w // self.patch_size
        return num_patches_h * self.patch_size, num_patches_w * self.patch_size

class Discriminator(nn.Module):
    def __init__(self, img_channels, discriminator_filters, discriminator_mlp_architecture):
        super(Discriminator, self).__init__()
        cnn_encoder_layers = []
        in_channels = img_channels
        for i, out_channels in enumerate(discriminator_filters):
            cnn_encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        self.cnn_encoder = nn.Sequential(*cnn_encoder_layers)
        self.last_cnn_out_channels = discriminator_filters[-1] if discriminator_filters else img_channels
        self.gap = nn.AdaptiveAvgPool2d(1)

        mlp_input_dim_discriminator = self.last_cnn_out_channels

        self.mlp = _build_mlp(
            mlp_input_dim_discriminator,
            discriminator_mlp_architecture,
            1
        )

    def forward(self, img):
        if not list(self.cnn_encoder.children()):
            encoded_map = img
        else:
            encoded_map = self.cnn_encoder(img)

        x_gap = self.gap(encoded_map)
        x_flattened = x_gap.view(x_gap.size(0), -1)
        output_logit = self.mlp(x_flattened)
        return output_logit

def _build_mlp(input_dim, hidden_layers_arch, output_dim, activation_fn_constructor=lambda: nn.LeakyReLU(0.2, inplace=True)):
    layers = []
    current_dim = input_dim
    if input_dim == 0:
        logging.warning("_build_mlp called with input_dim=0. This might be an issue. Setting to 1.")
        current_dim = 1

    for hidden_dim in hidden_layers_arch:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_fn_constructor())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def get_sobel_kernels(device):
    kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(IMG_CHANNELS, 1, 1, 1)
    kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(IMG_CHANNELS, 1, 1, 1)
    return kernel_x, kernel_y

def apply_sobel(img_batch, kernel_x, kernel_y, device):
    if kernel_x.device != img_batch.device: kernel_x = kernel_x.to(img_batch.device)
    if kernel_y.device != img_batch.device: kernel_y = kernel_y.to(img_batch.device)
    if img_batch.ndim == 3: img_batch = img_batch.unsqueeze(0)
    current_img_channels = img_batch.shape[1]
    if current_img_channels != kernel_x.shape[0]:
        k_x_base = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32, device=device)
        k_y_base = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32, device=device)
        kernel_x = k_x_base.view(1, 1, 3, 3).repeat(current_img_channels, 1, 1, 1).to(device)
        kernel_y = k_y_base.view(1, 1, 3, 3).repeat(current_img_channels, 1, 1, 1).to(device)
    sobel_padding = get_padding(3, 1, 1)
    edge_x = F.conv2d(img_batch, kernel_x, padding=sobel_padding, groups=current_img_channels)
    edge_y = F.conv2d(img_batch, kernel_y, padding=sobel_padding, groups=current_img_channels)
    return edge_x, edge_y

def compute_fft_magnitude_loss(pred_batch, target_batch, loss_fn):
    pred_batch_float = pred_batch.float()
    target_batch_float = target_batch.float()
    pred_fft = torch.fft.fft2(pred_batch_float, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.fft2(target_batch_float, dim=(-2, -1), norm="ortho")
    pred_fft_mag = torch.abs(pred_fft)
    target_fft_mag = torch.abs(target_fft)
    loss = loss_fn(pred_fft_mag, target_fft_mag)
    return loss

def find_latest_checkpoint(recordings_dir, model_name_prefix):
    checkpoint_dir = recordings_dir
    latest_epoch = -1
    latest_checkpoint_path = None
    pattern = re.compile(rf"^{re.escape(model_name_prefix)}_epoch_(\d+)\.pth$")
    if not os.path.isdir(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} not found.")
        return None, -1
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch :
                latest_epoch = epoch_num
                latest_checkpoint_path = os.path.join(checkpoint_dir, filename)
    if latest_checkpoint_path:
        logging.info(f"Found latest checkpoint: {latest_checkpoint_path} (Completed Epoch {latest_epoch})")
        return latest_checkpoint_path, latest_epoch -1
    else:
        logging.info(f"No checkpoint found for prefix '{model_name_prefix}' in '{checkpoint_dir}'.")
    return None, -1


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("torch.backends.cudnn.benchmark set to True.")

    pt_version = torch.__version__
    logging.info(f"PyTorch version: {pt_version}")
    use_torch_compile = False
    if pt_version.startswith("2."):
        logging.info(f"PyTorch version 2.x detected, torch.compile is available (currently {'enabled' if use_torch_compile else 'disabled'}).")
        if use_torch_compile and USE_GRADIENT_CHECKPOINTING:
            logging.warning("USE_GRADIENT_CHECKPOINTING is True, but torch.compile() might be incompatible. Disabling gradient checkpointing for compiled model.")
            USE_GRADIENT_CHECKPOINTING = False
    else:
        logging.info("PyTorch version < 2.0, torch.compile will not be used.")

    logging.info(f"Using device: {DEVICE}")
    logging.info(f"MODEL_SAVE_NAME: {MODEL_SAVE_NAME}")
    logging.info(f"ViT Config: Patch Size: {PATCH_SIZE}, Embed Dim: {EMBED_DIM}, Registrar Tokens: {NUM_REGISTRAR_TOKENS}")
    if USE_CONV_TOKENIZER:
        logging.info(f"  Using ConvTokenizer: Channels: {TOKENIZER_CONV_OUT_CHANNELS}, Num ResBlocks/Stage: {TOKENIZER_NUM_RESBLOCKS_PER_STAGE}")
    else:
        logging.info(f"  Using PatchEmbed (single Conv2d tokenizer).")
    logging.info(f"CNN Decoder Config: Filter Sizes (for ResBlocks): {DECODER_CNN_FILTER_SIZES}, Num ResBlocks/Stage: {DECODER_NUM_RESBLOCKS_PER_STAGE}")
    logging.info(f"Transformer Config: Heads: {TRANSFORMER_NHEAD}, Encoder Layers: {TRANSFORMER_NUM_ENCODER_LAYERS}")
    logging.info(f"  FFN Scale: {TRANSFORMER_DIM_FEEDFORWARD_SCALE}, Dropout: {TRANSFORMER_DROPOUT}, Interleaved Attn: {USE_INTERLEAVED_ATTENTION}")
    logging.info(f"Action Token Config: MLP Hidden: {ACTION_PROCESS_MLP_HIDDEN_LAYERS}")


    transform_for_model_input = transforms.Compose([
        transforms.Resize((INPUT_IMG_HEIGHT_GLOBAL, INPUT_IMG_WIDTH_GLOBAL)),
        transforms.ToTensor(),
    ])
    transform_pil_to_tensor_for_loss = transforms.ToTensor()


    dataloader_shuffle = True
    try:
        dataset = FramePredictionDataset(
            recordings_dir=RECORDINGS_DIR, log_filename=JSONL_LOG_FILENAME,
            img_transform=transform_for_model_input,
            dataloader_img_height=INPUT_IMG_HEIGHT_GLOBAL,
            dataloader_img_width=INPUT_IMG_WIDTH_GLOBAL,
            expected_controls_len=SENSOR_INPUT_DIM
        )
        if len(dataset) == 0: raise ValueError("Dataset is empty.")
        num_workers = 0
        persistent_workers = num_workers > 0
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=dataloader_shuffle,
            num_workers=num_workers, pin_memory=DEVICE.type == 'cuda',
            persistent_workers=persistent_workers, drop_last=True
        )
        logging.info(f"DataLoader: {len(dataset)} samples, {len(dataloader)} batches. Workers: {num_workers}, Persistent: {persistent_workers}, Drop last: True.")
    except Exception as e:
        logging.error(f"DataLoader setup failed: {e}", exc_info=True); exit()

    generator_base = None
    discriminator_base = None

    try:
        generator_base = TransformerNextFrameGenerator(
            img_channels=IMG_CHANNELS,
            sensor_dim=SENSOR_INPUT_DIM,
            embed_dim=EMBED_DIM,
            patch_size=PATCH_SIZE,
            nhead=TRANSFORMER_NHEAD,
            num_encoder_layers=TRANSFORMER_NUM_ENCODER_LAYERS,
            dim_feedforward_scale=TRANSFORMER_DIM_FEEDFORWARD_SCALE,
            dropout=TRANSFORMER_DROPOUT,
            input_img_height=INPUT_IMG_HEIGHT_GLOBAL,
            input_img_width=INPUT_IMG_WIDTH_GLOBAL,
            action_process_mlp_hidden_layers=ACTION_PROCESS_MLP_HIDDEN_LAYERS,
            decoder_cnn_filter_sizes=DECODER_CNN_FILTER_SIZES,
            transformer_bottleneck_noise_std_factor=TRANSFORMER_BOTTLENECK_NOISE_STD_FACTOR,
            use_interleaved_attention=USE_INTERLEAVED_ATTENTION,
            num_registrar_tokens=NUM_REGISTRAR_TOKENS,
            use_conv_tokenizer=USE_CONV_TOKENIZER,
            tokenizer_conv_out_channels=TOKENIZER_CONV_OUT_CHANNELS if USE_CONV_TOKENIZER else None,
            tokenizer_conv_kernel_size=TOKENIZER_CONV_KERNEL_SIZE if USE_CONV_TOKENIZER else 3,
            tokenizer_num_resblocks_per_stage=TOKENIZER_NUM_RESBLOCKS_PER_STAGE if USE_CONV_TOKENIZER else 1, # Pass new param
            decoder_num_resblocks_per_stage=DECODER_NUM_RESBLOCKS_PER_STAGE # Pass new param
        ).to(DEVICE)

        natural_out_h, natural_out_w = generator_base.calculate_natural_output_resolution(INPUT_IMG_HEIGHT_GLOBAL, INPUT_IMG_WIDTH_GLOBAL)
        logging.info(f"Generator's natural output resolution for {INPUT_IMG_HEIGHT_GLOBAL}x{INPUT_IMG_WIDTH_GLOBAL} input: {natural_out_h}x{natural_out_w}")
        if natural_out_h != INPUT_IMG_HEIGHT_GLOBAL or natural_out_w != INPUT_IMG_WIDTH_GLOBAL:
            logging.warning(f"Natural output HxW ({natural_out_h}x{natural_out_w}) differs from global input HxW ({INPUT_IMG_HEIGHT_GLOBAL}x{INPUT_IMG_WIDTH_GLOBAL}). Ensure loss targets are sized correctly.")


        if USE_GAN_TRAINING:
            discriminator_base = Discriminator(
                img_channels=IMG_CHANNELS,
                discriminator_filters=DISCRIMINATOR_FILTERS,
                discriminator_mlp_architecture=DISCRIMINATOR_MLP_ARCHITECTURE
            ).to(DEVICE)
            logging.info("Unconditional Discriminator initialized.")

        generator = generator_base
        discriminator = discriminator_base
        if use_torch_compile and DEVICE.type == 'cuda':
            logging.info("Attempting to compile Generator with torch.compile()...")
            try:
                generator = torch.compile(generator_base)
                logging.info("Generator compiled successfully.")
            except Exception as e:
                logging.warning(f"Failed to compile Generator: {e}. Using uncompiled model.")
                generator = generator_base
            if USE_GAN_TRAINING and discriminator_base is not None:
                logging.info("Attempting to compile Discriminator with torch.compile()...")
                try:
                    discriminator = torch.compile(discriminator_base)
                    logging.info("Discriminator compiled successfully.")
                except Exception as e:
                    logging.warning(f"Failed to compile Discriminator: {e}. Using uncompiled model.")
                    discriminator = discriminator_base
        else:
            logging.info("torch.compile() is not used.")

    except Exception as e:
        logging.error(f"Model initialization or compilation failed: {e}", exc_info=True); exit()

    total_params_gen = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Generator Parameters: {total_params_gen:,}")
    if USE_GAN_TRAINING and discriminator is not None:
        total_params_disc = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logging.info(f"Total Trainable Discriminator Parameters: {total_params_disc:,}")

    criterion_pixel = nn.L1Loss()
    criterion_sobel = nn.L1Loss()
    criterion_fft_mag = nn.L1Loss()
    criterion_adversarial = nn.BCEWithLogitsLoss()

    optimizer_G = optim.AdamW(generator.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer_D = None
    if USE_GAN_TRAINING and discriminator is not None:
        optimizer_D = optim.AdamW(discriminator.parameters(), lr=DISCRIMINATOR_LR, weight_decay=1e-5)

    scaler_G = GradScaler(enabled=(DEVICE.type == 'cuda'))
    scaler_D = GradScaler(enabled=(DEVICE.type == 'cuda' and USE_GAN_TRAINING and discriminator is not None))

    start_epoch_0_indexed = 0
    if RESUME_FROM_LATEST:
        latest_checkpoint_path, completed_epoch_0_indexed = find_latest_checkpoint(RECORDINGS_DIR, MODEL_SAVE_NAME)
        if latest_checkpoint_path:
            try:
                logging.info(f"Loading checkpoint: {latest_checkpoint_path}")
                checkpoint_data = torch.load(latest_checkpoint_path, map_location=DEVICE)
                compatible = True

                if checkpoint_data.get('model_architecture_type_saved', "unknown") != MODEL_ARCHITECTURE_TYPE:
                    logging.warning(f"Checkpoint model architecture type MISMATCH. Ckpt: {checkpoint_data.get('model_architecture_type_saved')}, Current: {MODEL_ARCHITECTURE_TYPE}. Starting fresh.")
                    compatible = False
                if compatible and checkpoint_data.get('model_save_name_ckpt', "N/A") != MODEL_SAVE_NAME:
                     logging.warning(f"Checkpoint model name prefix MISMATCH. Ckpt: {checkpoint_data.get('model_save_name_ckpt')}, Current: {MODEL_SAVE_NAME}. Consider starting fresh or ensure compatibility.")

                # Check ResNet block parameters
                if compatible and USE_CONV_TOKENIZER:
                    if checkpoint_data.get('tokenizer_num_resblocks_per_stage_saved', 1) != TOKENIZER_NUM_RESBLOCKS_PER_STAGE:
                        logging.warning(f"Checkpoint tokenizer_num_resblocks_per_stage mismatch. Ckpt: {checkpoint_data.get('tokenizer_num_resblocks_per_stage_saved',1)}, Current: {TOKENIZER_NUM_RESBLOCKS_PER_STAGE}. Starting fresh.")
                        compatible = False
                if compatible: # This check applies to all decoders now
                    if checkpoint_data.get('decoder_num_resblocks_per_stage_saved', 1) != DECODER_NUM_RESBLOCKS_PER_STAGE:
                        logging.warning(f"Checkpoint decoder_num_resblocks_per_stage mismatch. Ckpt: {checkpoint_data.get('decoder_num_resblocks_per_stage_saved',1)}, Current: {DECODER_NUM_RESBLOCKS_PER_STAGE}. Starting fresh.")
                        compatible = False

                if compatible and checkpoint_data.get('decoder_cnn_filter_sizes_saved', None) != DECODER_CNN_FILTER_SIZES:
                     logging.warning(f"Checkpoint decoder_cnn_filter_sizes mismatch. Ckpt: {checkpoint_data.get('decoder_cnn_filter_sizes_saved')}, Current: {DECODER_CNN_FILTER_SIZES}. Starting fresh.")
                     compatible = False

                ckpt_use_conv_tokenizer = checkpoint_data.get('use_conv_tokenizer_saved', False)
                if compatible and ckpt_use_conv_tokenizer != USE_CONV_TOKENIZER:
                    logging.warning(f"Checkpoint use_conv_tokenizer mismatch. Ckpt: {ckpt_use_conv_tokenizer}, Current: {USE_CONV_TOKENIZER}. Starting fresh.")
                    compatible = False

                if compatible and USE_CONV_TOKENIZER:
                    if checkpoint_data.get('tokenizer_conv_out_channels_saved', None) != TOKENIZER_CONV_OUT_CHANNELS:
                        logging.warning(f"Checkpoint tokenizer_conv_out_channels mismatch. Ckpt: {checkpoint_data.get('tokenizer_conv_out_channels_saved')}, Current: {TOKENIZER_CONV_OUT_CHANNELS}. Starting fresh.")
                        compatible = False
                    if checkpoint_data.get('tokenizer_conv_kernel_size_saved', None) != TOKENIZER_CONV_KERNEL_SIZE:
                        logging.warning(f"Checkpoint tokenizer_conv_kernel_size mismatch. Ckpt: {checkpoint_data.get('tokenizer_conv_kernel_size_saved')}, Current: {TOKENIZER_CONV_KERNEL_SIZE}. Starting fresh.")
                        compatible = False

                if compatible and checkpoint_data.get('patch_size_saved', -1) != PATCH_SIZE:
                     logging.warning(f"Checkpoint patch_size mismatch. Ckpt: {checkpoint_data.get('patch_size_saved')}, Current: {PATCH_SIZE}. Starting fresh.")
                     compatible = False
                if compatible and checkpoint_data.get('embed_dim_saved', -1) != EMBED_DIM:
                     logging.warning(f"Checkpoint embed_dim mismatch. Ckpt: {checkpoint_data.get('embed_dim_saved')}, Current: {EMBED_DIM}. Starting fresh.")
                     compatible = False
                if compatible and checkpoint_data.get('transformer_nhead_saved', -1) != TRANSFORMER_NHEAD:
                     logging.warning(f"Checkpoint transformer_nhead mismatch. Ckpt: {checkpoint_data.get('transformer_nhead_saved')}, Current: {TRANSFORMER_NHEAD}. Starting fresh.")
                     compatible = False
                if compatible and checkpoint_data.get('transformer_num_encoder_layers_saved', -1) != TRANSFORMER_NUM_ENCODER_LAYERS:
                     logging.warning(f"Checkpoint transformer_num_encoder_layers mismatch. Ckpt: {checkpoint_data.get('transformer_num_encoder_layers_saved')}, Current: {TRANSFORMER_NUM_ENCODER_LAYERS}. Starting fresh.")
                     compatible = False
                if compatible and checkpoint_data.get('use_interleaved_attention_saved', None) != USE_INTERLEAVED_ATTENTION:
                     logging.warning(f"Checkpoint use_interleaved_attention mismatch. Ckpt: {checkpoint_data.get('use_interleaved_attention_saved')}, Current: {USE_INTERLEAVED_ATTENTION}. Starting fresh.")
                     compatible = False
                if compatible and checkpoint_data.get('num_registrar_tokens_saved', 0) != NUM_REGISTRAR_TOKENS:
                     logging.warning(f"Checkpoint num_registrar_tokens mismatch. Ckpt: {checkpoint_data.get('num_registrar_tokens_saved', 0)}, Current: {NUM_REGISTRAR_TOKENS}. Starting fresh.")
                     compatible = False


                if compatible:
                    generator_to_load = generator._orig_mod if hasattr(generator, '_orig_mod') else generator
                    try:
                        generator_to_load.load_state_dict(checkpoint_data['generator_state_dict'])
                    except RuntimeError as e:
                        logging.warning(f"Error loading G state_dict: {e}. This might be due to architectural mismatch (e.g. ResNet changes). Forcing fresh start.")
                        compatible = False

                    if compatible:
                        optimizer_G.load_state_dict(checkpoint_data['optimizer_G_state_dict'])
                        if 'scaler_G_state_dict' in checkpoint_data and scaler_G.is_enabled() and checkpoint_data['scaler_G_state_dict']:
                            scaler_G.load_state_dict(checkpoint_data['scaler_G_state_dict'])

                        if USE_GAN_TRAINING and discriminator is not None:
                            if 'discriminator_state_dict' in checkpoint_data and 'optimizer_D_state_dict' in checkpoint_data:
                                discriminator_to_load = discriminator._orig_mod if hasattr(discriminator, '_orig_mod') else discriminator
                                try:
                                    discriminator_to_load.load_state_dict(checkpoint_data['discriminator_state_dict'])
                                    optimizer_D.load_state_dict(checkpoint_data['optimizer_D_state_dict'])
                                    if 'scaler_D_state_dict' in checkpoint_data and scaler_D.is_enabled() and checkpoint_data['scaler_D_state_dict']:
                                        scaler_D.load_state_dict(checkpoint_data['scaler_D_state_dict'])
                                except Exception as e_other_d:
                                     logging.warning(f"Could not load D states: {e_other_d}. Starting D fresh.")
                            else:
                                logging.warning("Checkpoint missing D states. Starting D fresh.")

                        saved_epoch_0_idx_from_ckpt_content = checkpoint_data.get('epoch', -1)
                        if completed_epoch_0_indexed != -1 and completed_epoch_0_indexed != saved_epoch_0_idx_from_ckpt_content:
                            logging.warning(f"Epoch mismatch! Filename implies completed epoch {completed_epoch_0_indexed}, Checkpoint content says {saved_epoch_0_idx_from_ckpt_content}. Using checkpoint content.")

                        start_epoch_0_indexed = saved_epoch_0_idx_from_ckpt_content + 1
                        logging.info(f"Resuming training from epoch {start_epoch_0_indexed + 1} (after completed epoch {saved_epoch_0_idx_from_ckpt_content +1}).")

                if not compatible:
                    logging.error("Checkpoint incompatible or load failed. Starting fresh.")
                    start_epoch_0_indexed = 0
            except Exception as e:
                logging.error(f"Error loading checkpoint {latest_checkpoint_path}: {e}. Starting fresh.", exc_info=True)
                start_epoch_0_indexed = 0
        else:
            logging.info(f"No compatible checkpoint found for '{MODEL_SAVE_NAME}'. Starting fresh.")
    else:
        logging.info("RESUME_FROM_LATEST is False. Starting fresh.")

    sobel_kernel_x_base, sobel_kernel_y_base = get_sobel_kernels(DEVICE)
    logging.info("--- Starting Training Loop ---")

    session_start_time = time.time()
    if start_epoch_0_indexed >= NUM_EPOCHS:
        logging.info(f"Target epochs ({NUM_EPOCHS}) already reached. Exiting."); exit()

    for epoch in range(start_epoch_0_indexed, NUM_EPOCHS):
        generator.train()
        if USE_GAN_TRAINING and discriminator is not None: discriminator.train()
        epoch_start_time = time.time()
        running_loss_recon_pixel, running_loss_pred_pixel, running_loss_sobel, running_loss_fft, running_loss_total_G = 0.0, 0.0, 0.0, 0.0, 0.0
        running_loss_D_real, running_loss_D_fake, running_loss_D_total_epoch, running_loss_G_adv = 0.0, 0.0, 0.0, 0.0
        processed_batches_G = 0
        processed_batches_D = 0

        for i, batch_data in enumerate(dataloader):
            if batch_data is None: logging.warning(f"Skipping None batch {i+1}."); continue
            try:
                current_img_paths_b, current_imgs_model_input_b, controls_b, \
                target_next_img_paths_b, _ = batch_data

                if not isinstance(current_img_paths_b, list) or not all(isinstance(p, str) for p in current_img_paths_b) or \
                   not isinstance(target_next_img_paths_b, list) or not all(isinstance(p, str) for p in target_next_img_paths_b):
                    logging.warning(f"Batch {i+1} has invalid image paths list format. Skipping.")
                    continue
                if any("ERR_" in p for p in current_img_paths_b) or any("ERR_" in p for p in target_next_img_paths_b):
                    logging.warning(f"Error paths detected in batch {i+1}. Skipping.")
                    continue
                if current_imgs_model_input_b.numel() == 0:
                    logging.warning(f"Skipping empty input image batch {i+1}."); continue
                if SENSOR_INPUT_DIM > 0 and controls_b.numel() == 0 :
                     logging.warning(f"Skipping empty controls batch {i+1} despite SENSOR_INPUT_DIM > 0."); continue
                elif SENSOR_INPUT_DIM == 0 and controls_b.numel() != 0 and controls_b.shape[1] != 0:
                     logging.warning(f"Controls batch {i+1} not (B,0) for SENSOR_INPUT_DIM = 0. Shape: {controls_b.shape}. Skipping."); continue
                if current_imgs_model_input_b.shape[0] != BATCH_SIZE:
                     logging.warning(f"Batch size mismatch in images for batch {i+1}. Expected {BATCH_SIZE}, got {current_imgs_model_input_b.shape[0]}. Skipping.")
                     continue
                if SENSOR_INPUT_DIM > 0 and controls_b.shape[0] != BATCH_SIZE:
                     logging.warning(f"Batch size mismatch in controls for batch {i+1}. Expected {BATCH_SIZE}, got {controls_b.shape[0]}. Skipping.")
                     continue
            except Exception as e:
                logging.error(f"Error unpacking or validating batch {i+1}: {e}", exc_info=True); continue

            current_imgs_model_input_b = current_imgs_model_input_b.to(DEVICE, non_blocking=True)
            controls_b = controls_b.to(DEVICE, non_blocking=True)

            if SENSOR_INPUT_DIM > 0:
                zero_actions = torch.zeros_like(controls_b).to(DEVICE)
            else:
                zero_actions = torch.empty(BATCH_SIZE, 0, device=DEVICE)

            target_recon_tensors_list = []
            target_pred_tensors_list = []
            valid_batch_targets = True
            try:
                target_h_for_loss, target_w_for_loss = natural_out_h, natural_out_w

                for path_str in current_img_paths_b:
                    pil_img = Image.open(path_str).convert('RGB')
                    tensor_img_orig_cpu = transform_pil_to_tensor_for_loss(pil_img)
                    resized_tensor_img = F.interpolate(tensor_img_orig_cpu.unsqueeze(0),
                                                       size=(target_h_for_loss, target_w_for_loss),
                                                       mode='bilinear', align_corners=False).squeeze(0)
                    target_recon_tensors_list.append(resized_tensor_img)
                resized_current_imgs_for_recon_loss = torch.stack(target_recon_tensors_list).to(DEVICE, non_blocking=True)

                for path_str in target_next_img_paths_b:
                    pil_img = Image.open(path_str).convert('RGB')
                    tensor_img_orig_cpu = transform_pil_to_tensor_for_loss(pil_img)
                    resized_tensor_img = F.interpolate(tensor_img_orig_cpu.unsqueeze(0),
                                                       size=(target_h_for_loss, target_w_for_loss),
                                                       mode='bilinear', align_corners=False).squeeze(0)
                    target_pred_tensors_list.append(resized_tensor_img)
                resized_target_next_imgs = torch.stack(target_pred_tensors_list).to(DEVICE, non_blocking=True)

                if resized_current_imgs_for_recon_loss.shape[0] != BATCH_SIZE or \
                   resized_target_next_imgs.shape[0] != BATCH_SIZE:
                    logging.warning(f"Batch {i+1} incomplete targets (expected {BATCH_SIZE}, got {resized_current_imgs_for_recon_loss.shape[0]}/{resized_target_next_imgs.shape[0]}). Skipping.")
                    valid_batch_targets = False
            except Exception as e:
                logging.error(f"Error loading/resizing targets for batch {i+1}: {e}. Skipping.", exc_info=True)
                valid_batch_targets = False
            if not valid_batch_targets: continue


            loss_D_total_iter = torch.tensor(0.0, device=DEVICE)
            loss_D_real_iter = torch.tensor(0.0, device=DEVICE)
            loss_D_fake_iter = torch.tensor(0.0, device=DEVICE)

            if USE_GAN_TRAINING and discriminator is not None:
                for d_step in range(D_TRAIN_RATIO):
                    optimizer_D.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=scaler_D.is_enabled()):
                        with torch.no_grad():
                             outputs_prediction_for_D = generator(current_imgs_model_input_b, controls_b).detach()

                        if USE_SOFT_LABELS: real_labels = torch.rand(BATCH_SIZE, 1, device=DEVICE) * (D_REAL_LABEL_HIGH - D_REAL_LABEL_LOW) + D_REAL_LABEL_LOW
                        else: real_labels = torch.full((BATCH_SIZE, 1), 0.0, device=DEVICE, dtype=torch.float32)
                        if USE_NOISY_LABELS: real_labels = torch.clamp(real_labels + (torch.randn_like(real_labels) * LABEL_NOISE_STRENGTH), 0.0, 1.0)

                        output_D_real = discriminator(resized_target_next_imgs)
                        loss_D_real_iter = criterion_adversarial(output_D_real, real_labels)

                        if USE_SOFT_LABELS: fake_labels = torch.rand(BATCH_SIZE, 1, device=DEVICE) * (D_FAKE_LABEL_HIGH - D_FAKE_LABEL_LOW) + D_FAKE_LABEL_LOW
                        else: fake_labels = torch.full((BATCH_SIZE, 1), 1.0, device=DEVICE, dtype=torch.float32)
                        if USE_NOISY_LABELS: fake_labels = torch.clamp(fake_labels + (torch.randn_like(fake_labels) * LABEL_NOISE_STRENGTH), 0.0, 1.0)

                        output_D_fake = discriminator(outputs_prediction_for_D)
                        loss_D_fake_iter = criterion_adversarial(output_D_fake, fake_labels)

                        loss_D_total_iter = loss_D_real_iter + loss_D_fake_iter

                    if not torch.isfinite(loss_D_total_iter):
                        logging.error(f"NaN/Inf D loss. Skipping D update. E{epoch+1} B{i+1} D_step{d_step+1}"); continue
                    scaler_D.scale(loss_D_total_iter).backward()
                    scaler_D.step(optimizer_D)
                    scaler_D.update()
                    running_loss_D_real += loss_D_real_iter.item()
                    running_loss_D_fake += loss_D_fake_iter.item()
                    running_loss_D_total_epoch += loss_D_total_iter.item()
                    processed_batches_D +=1


            optimizer_G.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=scaler_G.is_enabled()):
                outputs_prediction = generator(current_imgs_model_input_b, controls_b)
                loss_pixel_pred_val = criterion_pixel(outputs_prediction, resized_target_next_imgs)

                loss_pixel_recon_val = torch.tensor(0.0, device=DEVICE)
                if RECONSTRUCTION_LOSS_WEIGHT > 0:
                    outputs_reconstruction = generator(current_imgs_model_input_b, zero_actions)
                    loss_pixel_recon_val = criterion_pixel(outputs_reconstruction, resized_current_imgs_for_recon_loss)

                loss_sobel_val = torch.tensor(0.0, device=DEVICE)
                if SOBEL_WEIGHT > 0:
                    pred_edge_x, pred_edge_y = apply_sobel(outputs_prediction, sobel_kernel_x_base, sobel_kernel_y_base, DEVICE)
                    with torch.no_grad():
                        target_edge_x, target_edge_y = apply_sobel(resized_target_next_imgs.float(), sobel_kernel_x_base, sobel_kernel_y_base, DEVICE)
                    loss_sobel_val = criterion_sobel(pred_edge_x, target_edge_x) + criterion_sobel(pred_edge_y, target_edge_y)

                loss_fft_val = torch.tensor(0.0, device=DEVICE)
                if FFT_MAG_LOSS_WEIGHT > 0:
                    loss_fft_val = compute_fft_magnitude_loss(outputs_prediction, resized_target_next_imgs, criterion_fft_mag)

                loss_G_adv_val = torch.tensor(0.0, device=DEVICE)
                effective_adversarial_loss_weight = ADVERSARIAL_LOSS_WEIGHT if USE_GAN_TRAINING else 0.0

                if USE_GAN_TRAINING and discriminator is not None and ADVERSARIAL_LOSS_WEIGHT > 0:
                    output_D_fake_for_G = discriminator(outputs_prediction)

                    if USE_SOFT_LABELS: target_labels_for_G = torch.rand(BATCH_SIZE, 1, device=DEVICE) * (G_TARGET_LABEL_HIGH - G_TARGET_LABEL_LOW) + G_TARGET_LABEL_LOW
                    else: target_labels_for_G = torch.full((BATCH_SIZE, 1), 0.0, device=DEVICE, dtype=torch.float32)
                    if USE_NOISY_LABELS: target_labels_for_G = torch.clamp(target_labels_for_G + (torch.randn_like(target_labels_for_G) * LABEL_NOISE_STRENGTH), 0.0, 1.0)

                    loss_G_adv_val = criterion_adversarial(output_D_fake_for_G, target_labels_for_G)

                total_loss_G = (RECONSTRUCTION_LOSS_WEIGHT * loss_pixel_recon_val) + \
                               loss_pixel_pred_val + \
                               (SOBEL_WEIGHT * loss_sobel_val) + \
                               (FFT_MAG_LOSS_WEIGHT * loss_fft_val) + \
                               (effective_adversarial_loss_weight * loss_G_adv_val)

            if not torch.isfinite(total_loss_G):
                logging.error(f"NaN/Inf G loss. Skipping G update. E{epoch+1} B{i+1}"); continue
            scaler_G.scale(total_loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            running_loss_recon_pixel += loss_pixel_recon_val.item() if RECONSTRUCTION_LOSS_WEIGHT > 0 else 0.0
            running_loss_pred_pixel += loss_pixel_pred_val.item()
            running_loss_sobel += loss_sobel_val.item() if SOBEL_WEIGHT > 0 else 0.0
            running_loss_fft += loss_fft_val.item() if FFT_MAG_LOSS_WEIGHT > 0 else 0.0
            running_loss_G_adv += (effective_adversarial_loss_weight * loss_G_adv_val.item()) if USE_GAN_TRAINING and ADVERSARIAL_LOSS_WEIGHT > 0 else 0.0
            running_loss_total_G += total_loss_G.item()
            processed_batches_G += 1

            log_interval = max(1, len(dataloader) // 10) if len(dataloader) > 0 else 1
            if processed_batches_G > 0 and ((i + 1) % log_interval == 0 or (i + 1) == len(dataloader)):
                 current_scale_G = scaler_G.get_scale() if scaler_G.is_enabled() else 1.0
                 log_msg = (f'E[{epoch+1}/{NUM_EPOCHS}] B[{i+1}/{len(dataloader)}] '
                            f'G_Tot: {total_loss_G.item():.3f} (PRec: {loss_pixel_recon_val.item():.3f}, '
                            f'PPred: {loss_pixel_pred_val.item():.3f}, Sob: {loss_sobel_val.item():.3f}, '
                            f'FFT: {loss_fft_val.item():.3f}')
                 if USE_GAN_TRAINING and discriminator is not None:
                     current_scale_D = scaler_D.get_scale() if scaler_D.is_enabled() else 1.0
                     log_msg += (f', G_Adv: {loss_G_adv_val.item():.3f}(w:{effective_adversarial_loss_weight:.2f})) '
                                 f'D_Tot: {loss_D_total_iter.item():.3f} (D_R: {loss_D_real_iter.item():.3f}, D_F: {loss_D_fake_iter.item():.3f}) S_G:{current_scale_G:.0f} S_D:{current_scale_D:.0f}')
                 else:
                     log_msg += f') Scale_G: {current_scale_G:.0f}'
                 logging.info(log_msg)

        epoch_duration = time.time() - epoch_start_time
        avg_loss_total_G = running_loss_total_G / processed_batches_G if processed_batches_G > 0 else float('nan')
        avg_loss_recon_pixel = running_loss_recon_pixel / processed_batches_G if processed_batches_G > 0 else float('nan')
        avg_loss_pred_pixel = running_loss_pred_pixel / processed_batches_G if processed_batches_G > 0 else float('nan')
        avg_loss_sobel = running_loss_sobel / processed_batches_G if processed_batches_G > 0 else float('nan')
        avg_loss_fft = running_loss_fft / processed_batches_G if processed_batches_G > 0 else float('nan')
        avg_loss_G_adv = running_loss_G_adv / processed_batches_G if processed_batches_G > 0 else float('nan')

        summary_msg = (f"E{epoch+1} Sum | Time: {epoch_duration:.1f}s | AvgG_Tot: {avg_loss_total_G:.4f} "
                       f"(PRec: {avg_loss_recon_pixel:.4f}, PPred: {avg_loss_pred_pixel:.4f}, "
                       f"Sob: {avg_loss_sobel:.4f}, FFT: {avg_loss_fft:.4f}")
        if USE_GAN_TRAINING and discriminator is not None:
            avg_loss_D_total_epoch = running_loss_D_total_epoch / processed_batches_D if processed_batches_D > 0 else float('nan')
            avg_loss_D_real_epoch = running_loss_D_real / processed_batches_D if processed_batches_D > 0 else float('nan')
            avg_loss_D_fake_epoch = running_loss_D_fake / processed_batches_D if processed_batches_D > 0 else float('nan')
            summary_msg += (f", G_Adv: {avg_loss_G_adv:.4f}) "
                            f"AvgD_Tot: {avg_loss_D_total_epoch:.4f} (D_R: {avg_loss_D_real_epoch:.4f}, D_F: {avg_loss_D_fake_epoch:.4f})")
        else:
            summary_msg += ")"
        logging.info(summary_msg)

        save_interval = 10
        if (epoch + 1) % save_interval == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(RECORDINGS_DIR, f'{MODEL_SAVE_NAME}_epoch_{epoch+1}.pth')
            try:
                generator_state_to_save = generator._orig_mod.state_dict() if hasattr(generator, '_orig_mod') else generator.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'generator_state_dict': generator_state_to_save,
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'scaler_G_state_dict': scaler_G.state_dict() if scaler_G.is_enabled() else None,
                    'model_save_name_ckpt': MODEL_SAVE_NAME,
                    'model_architecture_type_saved': MODEL_ARCHITECTURE_TYPE,
                    'patch_size_saved': PATCH_SIZE,
                    'embed_dim_saved': EMBED_DIM,
                    'decoder_cnn_filter_sizes_saved': DECODER_CNN_FILTER_SIZES,
                    'transformer_nhead_saved': TRANSFORMER_NHEAD,
                    'transformer_num_encoder_layers_saved': TRANSFORMER_NUM_ENCODER_LAYERS,
                    'transformer_dim_feedforward_scale_saved': TRANSFORMER_DIM_FEEDFORWARD_SCALE,
                    'transformer_dropout_saved': TRANSFORMER_DROPOUT,
                    'use_interleaved_attention_saved': USE_INTERLEAVED_ATTENTION,
                    'num_registrar_tokens_saved': NUM_REGISTRAR_TOKENS,
                    'input_img_height_trained': INPUT_IMG_HEIGHT_GLOBAL,
                    'input_img_width_trained': INPUT_IMG_WIDTH_GLOBAL,
                    'sensor_dim_trained': SENSOR_INPUT_DIM,
                    'action_process_mlp_hidden_layers_saved': ACTION_PROCESS_MLP_HIDDEN_LAYERS,
                    'use_conv_tokenizer_saved': USE_CONV_TOKENIZER,
                    'decoder_num_resblocks_per_stage_saved': DECODER_NUM_RESBLOCKS_PER_STAGE,
                }
                if USE_CONV_TOKENIZER:
                    save_dict['tokenizer_conv_out_channels_saved'] = TOKENIZER_CONV_OUT_CHANNELS
                    save_dict['tokenizer_conv_kernel_size_saved'] = TOKENIZER_CONV_KERNEL_SIZE
                    save_dict['tokenizer_num_resblocks_per_stage_saved'] = TOKENIZER_NUM_RESBLOCKS_PER_STAGE


                if USE_GAN_TRAINING and discriminator is not None:
                    discriminator_state_to_save = discriminator._orig_mod.state_dict() if hasattr(discriminator, '_orig_mod') else discriminator.state_dict()
                    save_dict['discriminator_state_dict'] = discriminator_state_to_save
                    save_dict['optimizer_D_state_dict'] = optimizer_D.state_dict()
                    save_dict['scaler_D_state_dict'] = scaler_D.state_dict() if scaler_D.is_enabled() else None
                    save_dict['discriminator_filters_saved'] = DISCRIMINATOR_FILTERS
                    save_dict['discriminator_mlp_architecture_saved'] = DISCRIMINATOR_MLP_ARCHITECTURE
                torch.save(save_dict, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logging.error(f"Failed to save checkpoint {checkpoint_path}: {e}", exc_info=True)

    current_session_duration = time.time() - session_start_time
    logging.info(f"Training finished after {epoch+1} epochs.")
    logging.info(f"Total training session duration: {current_session_duration/3600:.2f} hours.")

    final_model_path = os.path.join(RECORDINGS_DIR, f'final_{MODEL_SAVE_NAME}_model.pth')
    try:
        if 'save_dict' not in locals() and 'epoch' in locals() and epoch >= 0 :
             generator_state_to_save = generator._orig_mod.state_dict() if hasattr(generator, '_orig_mod') else generator.state_dict()
             save_dict = {
                'epoch': epoch,
                'generator_state_dict': generator_state_to_save,
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'scaler_G_state_dict': scaler_G.state_dict() if scaler_G.is_enabled() else None,
                'model_save_name_ckpt': MODEL_SAVE_NAME,
                'model_architecture_type_saved': MODEL_ARCHITECTURE_TYPE,
                'patch_size_saved': PATCH_SIZE,
                'embed_dim_saved': EMBED_DIM,
                'decoder_cnn_filter_sizes_saved': DECODER_CNN_FILTER_SIZES,
                'transformer_nhead_saved': TRANSFORMER_NHEAD,
                'transformer_num_encoder_layers_saved': TRANSFORMER_NUM_ENCODER_LAYERS,
                'transformer_dim_feedforward_scale_saved': TRANSFORMER_DIM_FEEDFORWARD_SCALE,
                'transformer_dropout_saved': TRANSFORMER_DROPOUT,
                'use_interleaved_attention_saved': USE_INTERLEAVED_ATTENTION,
                'num_registrar_tokens_saved': NUM_REGISTRAR_TOKENS,
                'input_img_height_trained': INPUT_IMG_HEIGHT_GLOBAL,
                'input_img_width_trained': INPUT_IMG_WIDTH_GLOBAL,
                'sensor_dim_trained': SENSOR_INPUT_DIM,
                'action_process_mlp_hidden_layers_saved': ACTION_PROCESS_MLP_HIDDEN_LAYERS,
                'use_conv_tokenizer_saved': USE_CONV_TOKENIZER,
                'decoder_num_resblocks_per_stage_saved': DECODER_NUM_RESBLOCKS_PER_STAGE,
             }
             if USE_CONV_TOKENIZER:
                save_dict['tokenizer_conv_out_channels_saved'] = TOKENIZER_CONV_OUT_CHANNELS
                save_dict['tokenizer_conv_kernel_size_saved'] = TOKENIZER_CONV_KERNEL_SIZE
                save_dict['tokenizer_num_resblocks_per_stage_saved'] = TOKENIZER_NUM_RESBLOCKS_PER_STAGE
             if USE_GAN_TRAINING and discriminator is not None:
                discriminator_state_to_save = discriminator._orig_mod.state_dict() if hasattr(discriminator, '_orig_mod') else discriminator.state_dict()
                save_dict['discriminator_state_dict'] = discriminator_state_to_save
                save_dict['optimizer_D_state_dict'] = optimizer_D.state_dict()
                save_dict['scaler_D_state_dict'] = scaler_D.state_dict() if scaler_D.is_enabled() else None
                save_dict['discriminator_filters_saved'] = DISCRIMINATOR_FILTERS
                save_dict['discriminator_mlp_architecture_saved'] = DISCRIMINATOR_MLP_ARCHITECTURE


        if 'save_dict' in locals():
            final_save_dict = save_dict.copy()
            if 'epoch' in locals() and epoch >=0:
                final_save_dict['epoch'] = epoch
            torch.save(final_save_dict, final_model_path)
            logging.info(f"Final model (state_dict and key config) saved to {final_model_path}")
        else:
            logging.warning("No checkpoint was saved during training (or training didn't run long enough for a save). Final model not saved.")

    except Exception as e:
        logging.error(f"Failed to save final model {final_model_path}: {e}", exc_info=True)
