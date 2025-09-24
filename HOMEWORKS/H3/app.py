# -*- coding: utf-8 -*-
"""
Demonstrates bit-plane slicing and progressive image reconstruction.

This script performs the following actions:
1. Loads a grayscale image or creates a sample one if not found.
2. Decomposes the image into its 8 constituent bit planes.
3. Saves each bit plane as an individual image.
4. Progressively reconstructs the image by adding bit planes from MSB to LSB.
5. Saves an image at each step of the reconstruction.
6. Generates a video showing the gradual improvement in image quality.
7. Creates summary images comparing all bit planes and reconstruction steps.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# --- Configuration Constants ---
# Use Path objects for robust, cross-platform path handling.
SOURCE_IMAGE_PATH = Path("image.png")
OUTPUT_DIR = Path("images")
VIDEO_FILENAME = "bit_plane_progression.mp4"
VIDEO_FPS = 1  # We'll show each step for a fixed duration.
FRAME_DURATION_SECONDS = 2 # How long each step is visible in the video.

# --- Core Image Processing Functions ---

def extract_bit_plane(image: np.ndarray, bit_position: int) -> np.ndarray:
    """Extracts a specific bit plane from an 8-bit grayscale image.
    
    Args:
        image: The input 8-bit grayscale image as a NumPy array.
        bit_position: The bit to extract (0 for LSB, 7 for MSB).

    Returns:
        A binary image (values 0 or 255) representing the bit plane.
    """
    if not (0 <= bit_position <= 7):
        raise ValueError("Bit position must be between 0 and 7.")
    
    # Create a mask to isolate the desired bit.
    mask = 1 << bit_position
    
    # Use bitwise AND to get the bit, then shift it to the 0th position.
    bit_plane = (image & mask) >> bit_position
    
    # Scale to 0-255 for visualization.
    return (bit_plane * 255).astype(np.uint8)


def reconstruct_from_bit_planes(bit_planes: List[np.ndarray]) -> np.ndarray:
    """Reconstructs an image from a list of bit planes.
    
    Args:
        bit_planes: A list of 8 bit planes, starting from MSB (Bit 7) to LSB (Bit 0).

    Returns:
        The reconstructed 8-bit grayscale image.
    """
    if len(bit_planes) > 8:
        raise ValueError("Cannot reconstruct from more than 8 bit planes.")
        
    # Start with a black image of the same dimensions.
    reconstructed_image = np.zeros_like(bit_planes[0], dtype=np.uint8)
    
    # Add the contribution of each bit plane, weighted by its bit value.
    for i, plane in enumerate(bit_planes):
        # The weight is 2 to the power of the bit position (MSB is 7).
        weight = 2 ** (7 - i)
        # Convert plane from 0/255 back to 0/1 before multiplying.
        reconstructed_image += (plane // 255) * weight
        
    return reconstructed_image

# --- Helper & Utility Functions ---

def load_or_create_image(image_path: Path) -> np.ndarray:
    """Loads a grayscale image from the specified path or creates a sample if not found."""
    if image_path.exists():
        print(f"✅ Loading image from: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to load image from {image_path}.")
        return image
    else:
        print(f"⚠️ Image not found at '{image_path}'. Creating a sample image.")
        # Create a sample 512x512 image for demonstration.
        img = np.zeros((512, 512), dtype=np.uint8)
        # Add various shapes and gradients for interesting bit planes.
        cv2.rectangle(img, (50, 50), (200, 200), 128, -1)
        cv2.circle(img, (350, 256), 100, 220, -1)
        # Create a gradient
        for i in range(512):
            img[i, :] = np.clip(img[i, :] + (i / 4), 0, 255)
        cv2.imwrite(str(image_path), img)
        print(f"✅ Sample image saved as '{image_path}'.")
        return img

def add_text_overlay(image: np.ndarray, text: str) -> np.ndarray:
    """Adds a descriptive text overlay with a semi-transparent background to an image."""
    # Convert to BGR for color text.
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0) # Bright green
    
    # Add a black rectangle background for better readability.
    cv2.rectangle(output_image, (0, 0), (output_image.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(output_image, text, (10, 28), font, font_scale, color, thickness)
    
    return output_image

# --- Main Application ---

def main():
    """Main function to run the entire bit-plane analysis process."""
    print("🚀 Starting Bit-Plane Slicing Demonstration...")
    
    # 1. Setup
    OUTPUT_DIR.mkdir(exist_ok=True)
    original_image = load_or_create_image(SOURCE_IMAGE_PATH)
    cv2.imwrite(str(OUTPUT_DIR / "original_image.png"), original_image)

    # 2. Extract all 8 bit planes (from MSB to LSB)
    print("\nExtracting bit planes...")
    all_bit_planes = [extract_bit_plane(original_image, i) for i in range(7, -1, -1)]
    for i, plane in enumerate(all_bit_planes):
        bit_num = 7 - i
        cv2.imwrite(str(OUTPUT_DIR / f"bit_plane_{bit_num}.png"), plane)
    print("✅ All 8 bit planes extracted and saved.")

    # 3. Progressive reconstruction and video frame generation
    print("\nPerforming progressive reconstruction...")
    reconstructed_images = []
    video_frames = []
    
    for i in range(1, 9):
        # Use the first 'i' most significant bit planes for reconstruction.
        planes_to_use = all_bit_planes[:i]
        reconstructed = reconstruct_from_bit_planes(planes_to_use)
        reconstructed_images.append(reconstructed)
        
        # Save the image for the current step.
        cv2.imwrite(str(OUTPUT_DIR / f"reconstructed_step_{i}.png"), reconstructed)
        
        # Create a frame for the video with a text overlay.
        bit_names = ", ".join([str(j) for j in range(7, 7 - i, -1)])
        info_text = f"Step {i}/8: Using Bits [{bit_names}]"
        video_frames.append(add_text_overlay(reconstructed, info_text))
        
    print("✅ All 8 reconstruction steps saved.")
    
    # 4. Create the video
    print("\n🎬 Creating video...")
    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(VIDEO_FILENAME), fourcc, VIDEO_FPS, (width, height))
    
    for frame in video_frames:
        # Write each frame multiple times to control its duration on screen.
        for _ in range(FRAME_DURATION_SECONDS * VIDEO_FPS):
            video_writer.write(frame)
    
    video_writer.release()
    print(f"✅ Video saved as '{VIDEO_FILENAME}'")
    
    # 5. Create summary images using Matplotlib for layout
    print("\nCreating summary plots...")
    
    # Plot for individual bit planes
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    for i, ax in enumerate(axes.flat[1:]):
        if i < len(all_bit_planes):
            ax.imshow(all_bit_planes[i], cmap='gray')
            ax.set_title(f'Bit Plane {7-i} {"(MSB)" if i==0 else "(LSB)" if i==7 else ""}')
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle('Original Image and Individual Bit Planes', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "all_bit_planes.png")
    plt.close(fig)

    # Plot for progressive reconstruction
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, (ax, img) in enumerate(zip(axes.flat, reconstructed_images)):
        ax.imshow(img, cmap='gray')
        bit_names = ", ".join([str(j) for j in range(7, 7 - (i+1), -1)])
        ax.set_title(f'Step {i+1}: Bits [{bit_names}]')
        ax.axis('off')
    fig.suptitle('Progressive Image Reconstruction', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "progressive_reconstruction.png")
    plt.close(fig)
    print("✅ Summary plots saved.")

    # --- Final Summary ---
    print("\n" + "="*50)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
    print("="*50)
    print(f"All generated files can be found in the '{OUTPUT_DIR}/' directory.")
    print(f"The final video is named '{VIDEO_FILENAME}'.")
    print("\nYou can now compile your LaTeX report.")


if __name__ == "__main__":
    main()