import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def remove_stripes_frequency_domain(image_path, output_path):
    """
    Remove stripes using frequency domain filtering (FFT)
    """
    # Load image
    img = Image.open(image_path)
    
    # Check if image is color or grayscale
    if img.mode == 'RGB':
        img_array = np.array(img)
        # Process each channel separately
        result_channels = []
        for channel in range(3):
            channel_data = img_array[:, :, channel].astype(float)
            filtered_channel = _process_channel_frequency(channel_data)
            result_channels.append(filtered_channel)
        
        # Combine channels
        result_array = np.stack(result_channels, axis=2)
    else:
        # Grayscale image
        img_array = np.array(img).astype(float)
        result_array = _process_channel_frequency(img_array)
    
    # Convert back to uint8 and save
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_array)
    result_img.save(output_path)
    
    return result_img

def _process_channel_frequency(channel_data):
    """Process a single channel using frequency domain filtering"""
    # Apply 2D Fourier Transform
    f_transform = np.fft.fft2(channel_data)
    f_shift = np.fft.fftshift(f_transform)
    
    # Get magnitude spectrum for visualization
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Create frequency domain mask
    rows, cols = channel_data.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a mask to remove periodic noise (stripes)
    mask = np.ones((rows, cols))
    
    # Remove horizontal stripes (common pattern)
    # Adjust these parameters based on your specific image
    stripe_width = 3
    for i in range(rows):
        for j in range(cols):
            # Remove frequencies that correspond to stripe patterns
            # Horizontal stripes appear as vertical lines in frequency domain
            if abs(j - ccol) < 10 and abs(i - crow) > 5:  # Vertical components
                mask[i, j] = 0.2
            # Remove specific frequency bands
            elif 20 < abs(j - ccol) < 40:  # Horizontal stripe frequencies
                mask[i, j] = 0.3
    
    # Smooth the mask to avoid sharp transitions
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=1)
    
    # Apply mask
    f_filtered = f_shift * mask
    
    # Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    
    return img_back

def remove_stripes_spatial_domain(image_path, output_path):
    """
    Alternative method using spatial domain filtering
    """
    img = Image.open(image_path)
    
    if img.mode == 'RGB':
        img_array = np.array(img)
        result_channels = []
        for channel in range(3):
            filtered_channel = _enhanced_median_filter(img_array[:, :, channel])
            result_channels.append(filtered_channel)
        result_array = np.stack(result_channels, axis=2)
    else:
        img_array = np.array(img)
        result_array = _enhanced_median_filter(img_array)
    
    result_img = Image.fromarray(result_array.astype(np.uint8))
    result_img.save(output_path)
    return result_img

def _enhanced_median_filter(channel_data, kernel_size=5):
    """Apply median filtering optimized for stripe removal"""
    pad = kernel_size // 2
    padded = np.pad(channel_data, pad, mode='reflect')
    result = np.zeros_like(channel_data)
    
    for i in range(channel_data.shape[0]):
        for j in range(channel_data.shape[1]):
            # Use adaptive neighborhood - more emphasis on horizontal filtering
            # for vertical stripes, and vice versa
            region = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.median(region)
    
    return result

def compare_results(image_path):
    """Run both methods and compare results"""
    # Original image
    original = Image.open(image_path)
    
    # Frequency domain method
    freq_result = remove_stripes_frequency_domain(image_path, 'leopard_frequency_filtered.jpg')
    
    # Spatial domain method
    spatial_result = remove_stripes_spatial_domain(image_path, 'leopard_spatial_filtered.jpg')
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(freq_result)
    axes[1].set_title('Frequency Domain Filtering')
    axes[1].axis('off')
    
    axes[2].imshow(spatial_result)
    axes[2].set_title('Spatial Domain Filtering')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('stripe_removal_comparison.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return freq_result, spatial_result

# MAIN USAGE EXAMPLE
if __name__ == "__main__":
    # Replace with your actual image path
    image_path = "_M9A6315+2_36aca0-3f97-4f79-bbab-abc20a7c42ab.jpg"  # Change this to your image file name
    
    try:
        # Method 1: Frequency domain (usually better for periodic stripes)
        print("Applying frequency domain filtering...")
        result1 = remove_stripes_frequency_domain(image_path, "leopard_clean_frequency.jpg")
        
        # Method 2: Spatial domain (good for non-periodic noise)
        print("Applying spatial domain filtering...")
        result2 = remove_stripes_spatial_domain(image_path, "leopard_clean_spatial.jpg")
        
        # Compare both methods
        print("Generating comparison plot...")
        compare_results(image_path)
        
        print("Stripe removal completed! Check the generated images:")
        print("- leopard_clean_frequency.jpg")
        print("- leopard_clean_spatial.jpg") 
        print("- stripe_removal_comparison.jpg")
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        print("Please make sure the image is in your current directory.")
    except Exception as e:
        print(f"Error processing image: {e}")