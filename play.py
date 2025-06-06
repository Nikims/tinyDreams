# -*- coding: utf-8 -*-
import pygame
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
import random
import logging
import time
import re
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Import Model Definitions from worldmodel.py ---
try:
    from worldmodel import TransformerNextFrameGenerator, Discriminator, ResNetBlock, PatchEmbed, ConvTokenizer, CNNDecoderHead # Import all necessary classes
    logging.info("Successfully imported ViT model classes from worldmodel.py")
except ImportError as e:
    logging.error(f"Failed to import model classes from worldmodel.py: {e}")
    logging.error("Please ensure worldmodel.py contains the NEW ViT-based class definitions with CNN Decoder and ConvTokenizer options.")
    exit()

# --- Configuration ---
RECORDINGS_DIR = "recordings_n" # Make sure this matches your training script's output dir
JSONL_LOG_FILENAME = "recording_log.jsonl"

# --- IMPORTANT: Checkpoint Configuration ---
# This prefix MUST exactly match the MODEL_SAVE_NAME_BASE + suffixes from your training script
MODEL_FILENAME_PREFIX_FROM_TRAINING = "nextframe_predictor_vit_action_token_cnndec_convtok_res2_concat_reg12_decres2"
TARGET_EPOCH = 140 # The epoch of the checkpoint you want to load

# --- Configuration for Display Resolution ---
DISPLAY_IMG_WIDTH = 960
DISPLAY_IMG_HEIGHT = 720

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Pygame/Inference Settings ---
WINDOW_WIDTH = DISPLAY_IMG_WIDTH
WINDOW_HEIGHT = DISPLAY_IMG_HEIGHT
INPUT_SCALE_KEYBOARD = 50.0 # Target range -1 to 1 directly for WASD (was 20.0, 1.0 is more standard)
INPUT_SCALE_MOUSE = 2 # Adjusted sensitivity for mouse delta (was 4)
FPS = 100
WINDOW_TITLE = f"Interactive Predictor (ViT Action Token CNN Dec)"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Visualization Settings ---
VIS_BUTTON_SIZE = 40
VIS_BUTTON_PADDING = 10
VIS_BUTTON_ALPHA = 180
VIS_JOYSTICK_RADIUS = 40
VIS_JOYSTICK_INDICATOR_RADIUS = 10
VIS_ELEMENTS_START_X = 20
VIS_ELEMENTS_START_Y = 20

# --- Colors ---
WHITE, BLACK, GRAY, LIGHT_GRAY, GREEN, RED, BLUE = (255, 255, 255), (0, 0, 0), (100, 100, 100), (180, 180, 180), (0, 255, 0), (255, 0, 0), (0, 0, 255)
TRANSPARENT_GRAY, TRANSPARENT_GREEN, TRANSPARENT_RED, TRANSPARENT_BLUE = (100, 100, 100, VIS_BUTTON_ALPHA), (0, 255, 0, VIS_BUTTON_ALPHA), (255, 0, 0, VIS_BUTTON_ALPHA), (0, 0, 255, VIS_BUTTON_ALPHA)

MODEL_CLASS = TransformerNextFrameGenerator
DISCRIMINATOR_CLASS = Discriminator
# IMPORTANT: This MUST match the MODEL_ARCHITECTURE_TYPE in your training script
MODEL_ARCHITECTURE_TYPE_EXPECTED = "vit_transformer_cnndec_resnet"

def load_model(recordings_dir, model_filename_prefix, target_epoch, model_class_to_instantiate, discriminator_class_to_instantiate):
    checkpoint_filename = f"{model_filename_prefix}_epoch_{target_epoch}.pth"
    model_path = os.path.join(recordings_dir, checkpoint_filename)

    logging.info(f"Attempting to load ViT model checkpoint: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"CRITICAL: Model checkpoint file not found at: {model_path}")
        return None, None, None

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        logging.info(f"Loaded checkpoint from {model_path}.")
        checkpoint_model_name = checkpoint.get('model_save_name_ckpt', 'N/A')
        logging.info(f"  Checkpoint's original model name: {checkpoint_model_name}")

        checkpoint_arch_type = checkpoint.get('model_architecture_type_saved', 'unknown')
        if checkpoint_arch_type != MODEL_ARCHITECTURE_TYPE_EXPECTED:
            logging.error(f"CRITICAL: Checkpoint architecture type is '{checkpoint_arch_type}', but this script requires '{MODEL_ARCHITECTURE_TYPE_EXPECTED}'.")
            return None, None, None
        logging.info(f"Checkpoint architecture type '{checkpoint_arch_type}' matches expected '{MODEL_ARCHITECTURE_TYPE_EXPECTED}'.")

        generator_state_dict = checkpoint.get('generator_state_dict')
        if generator_state_dict is None:
            logging.error("Checkpoint invalid or missing 'generator_state_dict'.")
            return None, None, None
        discriminator_state_dict = checkpoint.get('discriminator_state_dict')

        # --- Extract ALL relevant parameters from checkpoint for instantiation ---
        img_channels_train = checkpoint.get('img_channels_trained', 3)
        sensor_dim_train = checkpoint.get('sensor_dim_trained', 4)
        input_img_h_train = checkpoint.get('input_img_height_trained', 256)
        input_img_w_train = checkpoint.get('input_img_width_trained', 256)

        embed_dim_train = checkpoint.get('embed_dim_saved', 512)
        patch_size_train = checkpoint.get('patch_size_saved', 16)
        transformer_nhead_train = checkpoint.get('transformer_nhead_saved', 8)
        num_encoder_layers_train = checkpoint.get('transformer_num_encoder_layers_saved', 16)
        dim_feedforward_scale_train = checkpoint.get('transformer_dim_feedforward_scale_saved', 4)
        transformer_dropout_train = checkpoint.get('transformer_dropout_saved', 0.0) # Use trained dropout for consistency, though often 0 for inference
        use_interleaved_attention_train = checkpoint.get('use_interleaved_attention_saved', False)
        action_mlp_hidden_train = checkpoint.get('action_process_mlp_hidden_layers_saved', [128, 256])
        num_registrar_tokens_train = checkpoint.get('num_registrar_tokens_saved', 0)
        transformer_bottleneck_noise_std_factor_train = checkpoint.get('transformer_bottleneck_noise_std_factor_saved', 0.0)

        decoder_cnn_filter_sizes_train = checkpoint.get('decoder_cnn_filter_sizes_saved', [128, 64, 32])
        decoder_num_resblocks_per_stage_train = checkpoint.get('decoder_num_resblocks_per_stage_saved', 1)

        use_conv_tokenizer_train = checkpoint.get('use_conv_tokenizer_saved', False)
        tokenizer_conv_out_channels_train = checkpoint.get('tokenizer_conv_out_channels_saved', None)
        tokenizer_conv_kernel_size_train = checkpoint.get('tokenizer_conv_kernel_size_saved', 3)
        tokenizer_num_resblocks_per_stage_train = checkpoint.get('tokenizer_num_resblocks_per_stage_saved', 1)

        # For inference, dropout and bottleneck noise are typically 0
        transformer_dropout_inference = 0.0
        transformer_bottleneck_noise_std_factor_inference = 0.0

        logging.info(f"--- Effective Configuration for ViT Model Instantiation (from Checkpoint) ---")
        logging.info(f"  img_channels: {img_channels_train}, sensor_dim: {sensor_dim_train}")
        logging.info(f"  Model TRAINED on input HxW: {input_img_h_train}x{input_img_w_train}")
        logging.info(f"  ViT Embed Dim: {embed_dim_train}, Patch Size: {patch_size_train}")
        if use_conv_tokenizer_train:
            logging.info(f"  Using ConvTokenizer: Channels: {tokenizer_conv_out_channels_train}, Kernel: {tokenizer_conv_kernel_size_train}, ResBlocks/Stage: {tokenizer_num_resblocks_per_stage_train}")
        else:
            logging.info(f"  Using PatchEmbed (single Conv2d tokenizer).")
        logging.info(f"  CNN Decoder Filters: {decoder_cnn_filter_sizes_train}, ResBlocks/Stage: {decoder_num_resblocks_per_stage_train}")
        logging.info(f"  Transformer Heads: {transformer_nhead_train}, Encoder Layers: {num_encoder_layers_train}")
        logging.info(f"  Interleaved Attention: {use_interleaved_attention_train}")
        logging.info(f"  Action MLP Hidden: {action_mlp_hidden_train}")
        logging.info(f"  Registrar Tokens: {num_registrar_tokens_train}")
        logging.info(f"  Bottleneck Noise Factor (Trained): {transformer_bottleneck_noise_std_factor_train} (Inference: {transformer_bottleneck_noise_std_factor_inference})")
        logging.info("--------------------------------------------------------------------")

        generator_kwargs = {
            'img_channels': img_channels_train,
            'sensor_dim': sensor_dim_train,
            'embed_dim': embed_dim_train,
            'patch_size': patch_size_train,
            'nhead': transformer_nhead_train,
            'num_encoder_layers': num_encoder_layers_train,
            'dim_feedforward_scale': dim_feedforward_scale_train,
            'dropout': transformer_dropout_inference, # Use 0.0 for inference
            'input_img_height': input_img_h_train,
            'input_img_width': input_img_w_train,
            'action_process_mlp_hidden_layers': action_mlp_hidden_train,
            'decoder_cnn_filter_sizes': decoder_cnn_filter_sizes_train,
            'transformer_bottleneck_noise_std_factor': transformer_bottleneck_noise_std_factor_inference, # Use 0.0 for inference
            'use_interleaved_attention': use_interleaved_attention_train,
            'num_registrar_tokens': num_registrar_tokens_train,
            'use_conv_tokenizer': use_conv_tokenizer_train,
            'tokenizer_conv_out_channels': tokenizer_conv_out_channels_train,
            'tokenizer_conv_kernel_size': tokenizer_conv_kernel_size_train,
            'tokenizer_num_resblocks_per_stage': tokenizer_num_resblocks_per_stage_train,
            'decoder_num_resblocks_per_stage': decoder_num_resblocks_per_stage_train
        }

        logging.info(f"--- Instantiating Generator ({model_class_to_instantiate.__name__}) ---")
        model_G_base = model_class_to_instantiate(**generator_kwargs).to(DEVICE)

        # Remove 'module.' prefix if model was saved with DataParallel
        new_generator_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in generator_state_dict.items() }

        try:
            incompatible_keys_g = model_G_base.load_state_dict(new_generator_state_dict, strict=True)
            if incompatible_keys_g.missing_keys or incompatible_keys_g.unexpected_keys:
                logging.warning(f"Generator state_dict loaded with warnings: Missing keys: {incompatible_keys_g.missing_keys}, Unexpected keys: {incompatible_keys_g.unexpected_keys}")
            else:
                logging.info("Generator state_dict loaded successfully (strict=True).")
        except RuntimeError as e:
             logging.error(f"RuntimeError loading Generator state_dict: {e}. This often indicates a mismatch in model architecture. Exiting.")
             return None, None, None

        model_G_base.eval()
        logging.info(f"Generator '{model_class_to_instantiate.__name__}' loaded to {DEVICE} in eval mode.")
        generator = model_G_base

        model_D_base = None
        discriminator = None
        use_gan_training_in_ckpt = checkpoint.get('use_gan_training', False) # Check if GAN was used in training
        if use_gan_training_in_ckpt and discriminator_state_dict is not None:
            discriminator_filters_train = checkpoint.get('discriminator_filters_saved', [32, 64, 128])
            discriminator_mlp_architecture_train = checkpoint.get('discriminator_mlp_architecture_saved', [512, 256, 1])
            logging.info(f"--- Instantiating Discriminator ({discriminator_class_to_instantiate.__name__}) ---")
            try:
                model_D_base = discriminator_class_to_instantiate(
                    img_channels=img_channels_train,
                    discriminator_filters=discriminator_filters_train,
                    discriminator_mlp_architecture=discriminator_mlp_architecture_train,
                ).to(DEVICE)
                new_discriminator_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in discriminator_state_dict.items() }
                model_D_base.load_state_dict(new_discriminator_state_dict, strict=True)
                model_D_base.eval()
                discriminator = model_D_base
                logging.info(f"Discriminator loaded to {DEVICE} in eval mode.")
            except Exception as e_d:
                logging.warning(f"Could not load Discriminator: {e_d}. Discriminator will not be used.")
                discriminator = None
        else:
            logging.info("Discriminator not loaded (GAN not used in training or state_dict missing).")

        model_actual_output_h, model_actual_output_w = generator.calculate_natural_output_resolution(input_img_h_train, input_img_w_train)
        logging.info(f"Model's natural output resolution for {input_img_h_train}x{input_img_w_train} input: {model_actual_output_h}x{model_actual_output_w}")

        final_loaded_config_for_inference = {
            'model_trained_input_h': input_img_h_train,
            'model_trained_input_w': input_img_w_train,
            'model_native_output_h': model_actual_output_h,
            'model_native_output_w': model_actual_output_w,
            'img_channels': img_channels_train,
            'sensor_dim': sensor_dim_train,
            'use_gan_training': use_gan_training_in_ckpt and discriminator is not None
        }
        return generator, discriminator, final_loaded_config_for_inference

    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None, None, None

def tensor_to_surface(tensor, target_width, target_height):
    if tensor is None:
        surface = pygame.Surface((target_width, target_height)); surface.fill((127,127,127)); return surface
    try:
        tensor = tensor.detach().cpu()
        tensor = torch.clamp(tensor, 0.0, 1.0)
        img_np = tensor.numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

        if img_np.shape[2] == 1: img_np = np.repeat(img_np, 3, axis=2)
        img_np_contiguous = np.ascontiguousarray(img_np)
        temp_surface = pygame.image.frombuffer(img_np_contiguous.tobytes(), (img_np.shape[1], img_np.shape[0]), 'RGB')

        if temp_surface.get_width() != target_width or temp_surface.get_height() != target_height:
            return pygame.transform.smoothscale(temp_surface, (target_width, target_height))
        return temp_surface
    except Exception as e:
        logging.error(f"Error converting tensor to surface: {e}", exc_info=True)
        surface = pygame.Surface((target_width, target_height)); surface.fill((0,127,0)); return surface

def get_valid_image_paths(recordings_dir, log_filename):
    log_path = os.path.join(recordings_dir, log_filename)
    valid_image_files = []
    if not os.path.exists(log_path):
        logging.warning(f"Log file {log_path} not found for initial frame. Will use gray frame.")
        return []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    filename = entry.get("frame_filename")
                    if filename:
                        full_path = os.path.join(recordings_dir, filename)
                        if os.path.isfile(full_path): valid_image_files.append(full_path)
                except json.JSONDecodeError: continue
    except Exception as e:
        logging.error(f"Error reading log file {log_path}: {e}")
    logging.info(f"Found {len(valid_image_files)} valid image paths in log.")
    return valid_image_files

def load_initial_frame(valid_image_paths, transform_for_model_input, target_img_channels, target_img_height, target_img_width):
    expected_shape = (target_img_channels, target_img_height, target_img_width)
    fallback_tensor = torch.full(expected_shape, 0.5, dtype=torch.float32).to(DEVICE)

    if not valid_image_paths:
        logging.warning("No valid image paths for initial frame. Using gray fallback.")
        return fallback_tensor

    random_paths_to_try = random.sample(valid_image_paths, min(10, len(valid_image_paths)))
    for random_image_path in random_paths_to_try:
        try:
            img = Image.open(random_image_path).convert('RGB')
            initial_tensor = transform_for_model_input(img)
            logging.info(f"Loaded initial frame: {os.path.basename(random_image_path)} with shape {initial_tensor.shape} (Target: {target_img_height}x{target_img_width})")
            return initial_tensor.to(DEVICE)
        except Exception as e:
            logging.error(f"Error loading/transforming initial image {random_image_path}: {e}.")
    logging.error(f"Failed to load any valid initial frame. Falling back to gray frame.")
    return fallback_tensor

def run_inference():
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    try: font = pygame.font.Font(None, 28)
    except pygame.error: font = None; logging.warning("Font not loaded.")

    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    model_G, model_D, loaded_config = load_model(
        RECORDINGS_DIR, MODEL_FILENAME_PREFIX_FROM_TRAINING, TARGET_EPOCH,
        MODEL_CLASS, DISCRIMINATOR_CLASS
    )

    if model_G is None or loaded_config is None:
        logging.error("Failed to load model. Exiting."); pygame.quit(); return

    model_trained_input_h = loaded_config['model_trained_input_h']
    model_trained_input_w = loaded_config['model_trained_input_w']
    model_native_output_h = loaded_config['model_native_output_h']
    model_native_output_w = loaded_config['model_native_output_w']
    img_c = loaded_config['img_channels']
    model_sensor_dim = loaded_config['sensor_dim']
    use_discriminator = loaded_config['use_gan_training']

    pygame.display.set_caption(f"{WINDOW_TITLE} - {MODEL_FILENAME_PREFIX_FROM_TRAINING}_ep{TARGET_EPOCH} (In: {model_trained_input_h}x{model_trained_input_w} Out: {model_native_output_h}x{model_native_output_w})")

    transform_predicted_to_next_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((model_trained_input_h, model_trained_input_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    initial_frame_disk_transform = transforms.Compose([
        transforms.Resize((model_trained_input_h, model_trained_input_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    valid_image_paths = get_valid_image_paths(RECORDINGS_DIR, JSONL_LOG_FILENAME)
    current_input_frame_tensor = load_initial_frame(
        valid_image_paths, initial_frame_disk_transform,
        img_c, model_trained_input_h, model_trained_input_w
    )

    left_stick_x, left_stick_y = 0.0, 0.0
    right_stick_x, right_stick_y = 0.0, 0.0
    running = True; frame_count = 0; total_inference_time_ms = 0; total_display_time_ms = 0; total_disc_time_ms = 0
    start_loop_time = time.perf_counter()
    inverted_plausibility = float('nan')
    vis_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

    logging.info(f"Starting inference loop. Device: {DEVICE}. Model input: {model_trained_input_h}x{model_trained_input_w}, Model output: {model_native_output_h}x{model_native_output_w}.")

    while running:
        mouse_dx_raw, mouse_dy_raw = 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                current_input_frame_tensor = load_initial_frame(
                    valid_image_paths, initial_frame_disk_transform,
                    img_c, model_trained_input_h, model_trained_input_w
                )
            if event.type == pygame.MOUSEMOTION:
                mouse_dx_raw, mouse_dy_raw = event.rel
        if not running: break

        keys = pygame.key.get_pressed()
        target_ls_y = float(keys[pygame.K_w] - keys[pygame.K_s]) * INPUT_SCALE_KEYBOARD
        target_ls_x = float(keys[pygame.K_d] - keys[pygame.K_a]) * INPUT_SCALE_KEYBOARD
        target_rs_x = mouse_dx_raw * INPUT_SCALE_MOUSE
        target_rs_y = -mouse_dy_raw * INPUT_SCALE_MOUSE # Pygame y is inverted

        lerp_factor = 0.4
        left_stick_x = (1 - lerp_factor) * left_stick_x + lerp_factor * target_ls_x
        left_stick_y = (1 - lerp_factor) * left_stick_y + lerp_factor * target_ls_y
        right_stick_x = (1 - lerp_factor) * right_stick_x + lerp_factor * target_rs_x
        right_stick_y = (1 - lerp_factor) * right_stick_y + lerp_factor * target_rs_y

        final_controls_list = [left_stick_x, left_stick_y, right_stick_x, right_stick_y]
        if model_sensor_dim > 0:
            actions_tensor = torch.tensor(final_controls_list[:model_sensor_dim], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            actions_tensor = torch.empty(1, 0, dtype=torch.float32).to(DEVICE)

        current_input_frame_batch = current_input_frame_tensor.unsqueeze(0)

        inference_start_time = time.perf_counter()
        predicted_next_frame_tensor = None
        try:
            with torch.no_grad():
                predicted_next_frame_tensor_batch = model_G(current_input_frame_batch, actions_tensor)
                predicted_next_frame_tensor = predicted_next_frame_tensor_batch.squeeze(0)
        except Exception as e:
            logging.error(f"Error during model inference: {e}", exc_info=True)
            predicted_next_frame_tensor = torch.full((img_c, model_native_output_h, model_native_output_w), 0.5, dtype=torch.float32).to(DEVICE)
        total_inference_time_ms += (time.perf_counter() - inference_start_time) * 1000

        if use_discriminator and model_D is not None:
            disc_start_time = time.perf_counter()
            try:
                with torch.no_grad():
                    discriminator_output_logits = model_D(predicted_next_frame_tensor.unsqueeze(0))
                    discriminator_plausibility_score = torch.sigmoid(discriminator_output_logits).item()
                    inverted_plausibility = 1.0 - discriminator_plausibility_score
            except Exception as e:
                logging.error(f"Error during discriminator inference: {e}"); inverted_plausibility = float('nan')
            total_disc_time_ms += (time.perf_counter() - disc_start_time) * 1000

        display_start_time = time.perf_counter()
        current_window_width, current_window_height = screen.get_size()
        display_surface = tensor_to_surface(predicted_next_frame_tensor, current_window_width, current_window_height)
        screen.fill(BLACK)
        screen.blit(display_surface, (0, 0))

        vis_surface.fill((0, 0, 0, 0))
        button_y_start = VIS_ELEMENTS_START_Y
        button_x_start = VIS_ELEMENTS_START_X
        button_size = VIS_BUTTON_SIZE
        padding = VIS_BUTTON_PADDING

        w_color = TRANSPARENT_GREEN if keys[pygame.K_w] else TRANSPARENT_GRAY
        pygame.draw.rect(vis_surface, w_color, (button_x_start + button_size + padding, button_y_start, button_size, button_size), 0, 5)
        if font: text_surface = font.render("W", True, BLACK); text_rect = text_surface.get_rect(center=(button_x_start + button_size + padding + button_size//2, button_y_start + button_size//2)); vis_surface.blit(text_surface, text_rect)
        a_color = TRANSPARENT_GREEN if keys[pygame.K_a] else TRANSPARENT_GRAY
        pygame.draw.rect(vis_surface, a_color, (button_x_start, button_y_start + button_size + padding, button_size, button_size), 0, 5)
        if font: text_surface = font.render("A", True, BLACK); text_rect = text_surface.get_rect(center=(button_x_start + button_size//2, button_y_start + button_size + padding + button_size//2)); vis_surface.blit(text_surface, text_rect)
        s_color = TRANSPARENT_GREEN if keys[pygame.K_s] else TRANSPARENT_GRAY
        pygame.draw.rect(vis_surface, s_color, (button_x_start + button_size + padding, button_y_start + button_size + padding, button_size, button_size), 0, 5)
        if font: text_surface = font.render("S", True, BLACK); text_rect = text_surface.get_rect(center=(button_x_start + button_size + padding + button_size//2, button_y_start + button_size + padding + button_size//2)); vis_surface.blit(text_surface, text_rect)
        d_color = TRANSPARENT_GREEN if keys[pygame.K_d] else TRANSPARENT_GRAY
        pygame.draw.rect(vis_surface, d_color, (button_x_start + 2*(button_size + padding), button_y_start + button_size + padding, button_size, button_size), 0, 5)
        if font: text_surface = font.render("D", True, BLACK); text_rect = text_surface.get_rect(center=(button_x_start + 2*(button_size + padding) + button_size//2, button_y_start + button_size + padding + button_size//2)); vis_surface.blit(text_surface, text_rect)

        joystick_center_x = button_x_start + button_size + padding + button_size//2
        joystick_center_y = button_y_start + button_size + padding + button_size//2 + button_size + padding + VIS_JOYSTICK_RADIUS + padding
        joystick_radius = VIS_JOYSTICK_RADIUS
        indicator_radius = VIS_JOYSTICK_INDICATOR_RADIUS
        pygame.draw.circle(vis_surface, TRANSPARENT_GRAY, (joystick_center_x, joystick_center_y), joystick_radius, 0)

        # Left stick visualization (normalized to [-1, 1] as INPUT_SCALE_KEYBOARD is 1.0)
        norm_ls_x = left_stick_x
        norm_ls_y = left_stick_y
        indicator_pos_x = joystick_center_x + int(norm_ls_x * (joystick_radius - indicator_radius))
        indicator_pos_y = joystick_center_y + int(norm_ls_y * (joystick_radius - indicator_radius)) # pygame y is already correct for display
        pygame.draw.circle(vis_surface, TRANSPARENT_BLUE, (indicator_pos_x, indicator_pos_y), indicator_radius, 0)
        if font: text_surface = font.render("LS", True, BLACK); text_rect = text_surface.get_rect(center=(joystick_center_x, joystick_center_y - joystick_radius - padding)); vis_surface.blit(text_surface, text_rect)

        r_joystick_center_x = joystick_center_x + joystick_radius * 2 + padding * 2
        r_joystick_center_y = joystick_center_y
        pygame.draw.circle(vis_surface, TRANSPARENT_GRAY, (r_joystick_center_x, r_joystick_center_y), joystick_radius, 0)

        # Right stick visualization (mouse) - normalize based on a heuristic max visual delta
        max_mouse_delta_vis_scaled = 100.0 # Heuristic max for right_stick_x/y after INPUT_SCALE_MOUSE
        norm_rs_x = right_stick_x / max_mouse_delta_vis_scaled
        norm_rs_y = right_stick_y / max_mouse_delta_vis_scaled
        r_indicator_pos_x = r_joystick_center_x + int(norm_rs_x * (joystick_radius - indicator_radius))
        r_indicator_pos_y = r_joystick_center_y + int(norm_rs_y * (joystick_radius - indicator_radius)) # pygame y is already correct
        pygame.draw.circle(vis_surface, TRANSPARENT_RED, (r_indicator_pos_x, r_indicator_pos_y), indicator_radius, 0)
        if font: text_surface = font.render("RS", True, BLACK); text_rect = text_surface.get_rect(center=(r_joystick_center_x, r_joystick_center_y - joystick_radius - padding)); vis_surface.blit(text_surface, text_rect)

        if use_discriminator and font:
            plausibility_text = f"Plausibility: {inverted_plausibility:.2f}" if not math.isnan(inverted_plausibility) else "Plausibility: N/A"
            text_surface = font.render(plausibility_text, True, WHITE)
            text_rect = text_surface.get_rect(topright=(current_window_width - VIS_ELEMENTS_START_X, VIS_ELEMENTS_START_Y))
            screen.blit(text_surface, text_rect)

        screen.blit(vis_surface, (0,0))
        total_display_time_ms += (time.perf_counter() - display_start_time) * 1000
        pygame.display.flip()

        if predicted_next_frame_tensor is not None:
            resized_tensor_for_next_input = transform_predicted_to_next_input(predicted_next_frame_tensor.cpu())
            current_input_frame_tensor = resized_tensor_for_next_input.to(DEVICE).detach()

        frame_count += 1
        if frame_count > 0 and frame_count % (int(FPS * 2)) == 0:
             avg_loop_fps = frame_count / (time.perf_counter() - start_loop_time)
             avg_inf_time = total_inference_time_ms / frame_count
             avg_disc_time_str = f"{total_disc_time_ms / frame_count:.1f}ms" if total_disc_time_ms > 0 else "N/A"
             log_msg = (f"PygameFPS:{clock.get_fps():.1f} | AvgLoopFPS:{avg_loop_fps:.1f} | "
                        f"AvgG_Inf:{avg_inf_time:.1f}ms | AvgD_Inf:{avg_disc_time_str}")
             logging.info(log_msg)
        clock.tick(FPS)

    logging.info("Exiting inference loop.")
    pygame.quit()

if __name__ == "__main__":
    logging.info(f"Attempting to load checkpoint with prefix: '{MODEL_FILENAME_PREFIX_FROM_TRAINING}' at epoch {TARGET_EPOCH} from dir '{RECORDINGS_DIR}'")
    run_inference()