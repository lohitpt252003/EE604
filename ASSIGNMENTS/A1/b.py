import cv2
import numpy as np
from matplotlib import pyplot as plt

def remove_leopard_spots(image_path):
    """
    Removes 'leopard spots' (dark regions) from an image using HSV color segmentation
    and inpainting techniques.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        None: Displays the original, mask, and inpainted images.
    """
    # --- Image Loading and Preprocessing ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert to RGB for matplotlib display later
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Step 1: Create a mask for the spots ---
    # This is the trickiest part and will likely require fine-tuning.
    # We're looking for dark brown/black spots.
    # Define a range for dark colors in HSV. These values are approximate
    # and might need adjustment based on the specific image.

    # Lower bound for dark brown/black (adjust these!)
    lower_spot_hsv = np.array([0, 0, 0]) # Hue, Saturation, Value (V = brightness)
    upper_spot_hsv = np.array([180, 255, 70]) # Max Hue, Saturation, a relatively low Value for darkness

    # Create a mask for the spots
    spot_mask = cv2.inRange(hsv, lower_spot_hsv, upper_spot_hsv)

    # Refine the mask with morphological operations
    # Dilation to make spots slightly larger for inpainting (ensures edges are covered)
    kernel = np.ones((3,3), np.uint8)
    spot_mask = cv2.dilate(spot_mask, kernel, iterations=1)
    # Erosion to remove small noise, if any (uncomment if needed)
    # spot_mask = cv2.erode(spot_mask, kernel, iterations=1)

    # --- Step 2: Use inpainting to fill the masked areas ---
    # Inpainting reconstructs the masked area from the surrounding pixels.
    # For natural images, INPAINT_TELEA often gives good results.
    inpainted_img = cv2.inpaint(img, spot_mask, 3, cv2.INPAINT_TELEA)

    # Convert inpainted image to RGB for display
    inpainted_img_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)

    # --- Display Results ---
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(spot_mask, cmap='gray')
    plt.title('Identified Spots Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(inpainted_img_rgb)
    plt.title('Spots Removed (Inpainted)')
    plt.axis('off')

    plt.show()

    # --- Optional: Save the Result ---
    # cv2.imwrite('leopard_spots_removed.jpg', inpainted_img)
    # print("Result saved as 'leopard_spots_removed.jpg'")


if __name__ == '__main__':
    # --- Example Usage ---
    # Save your provided image as 'image.jpg' in the same directory as this script,
    # or provide the full path to your image file.
    remove_leopard_spots('image.jpg')