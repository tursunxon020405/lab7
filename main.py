import cv2
import numpy as np
import matplotlib.pyplot as plt

def single_scale_retinex(img, sigma):
    img = img.astype(np.float32) + 1.0
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigmaX=sigma))
    return retinex
#Bu uchinchi yulash uchun yozilgan izoh
#bu 4-yuklash uchun yozilgan izoh
def multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    retinex /= len(sigmas)
    return retinex

def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    img_sum[img_sum == 0] = 1  # Avoid division by zero
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

def simplest_color_balance(img, low_clip, high_clip):
    result = np.zeros_like(img)
    for i in range(img.shape[2]):  # Color balance each channel
        channel = img[:, :, i]
        low_val = np.percentile(channel, low_clip * 100)
        high_val = np.percentile(channel, high_clip * 100)
        channel = np.clip(channel, low_val, high_val)
        channel = (channel - low_val) / (high_val - low_val) * 255
        result[:, :, i] = channel
    return result.astype(np.uint8)

def msrcr_enhancement(image_path, sigmas=[15, 80, 250], G=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found. Please check the path.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) + 1.0

    retinex = multi_scale_retinex(img, sigmas)
    color_rest = color_restoration(img, alpha, beta)
    msrcr_result = G * (retinex * color_rest + b)

    msrcr_result = np.clip(msrcr_result, 0, 255)

    balanced_img = simplest_color_balance(msrcr_result.astype(np.uint8), low_clip, high_clip)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("MSRCR Enhanced Image")
    plt.imshow(balanced_img)
    plt.axis('off')

    plt.show()

# Run enhancement on your image
if __name__ == "__main__":
    image_path = "aslrasm.jpg"  # <-- Change this to your image file path
    msrcr_enhancement(image_path)
