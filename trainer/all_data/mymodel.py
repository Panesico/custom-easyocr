import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.splits = ['train', 'test', 'val']
        self.data = {}
        for split in self.splits:
            split_path = self.base_path / split
            csv_file = list(split_path.glob('*.csv'))[0]
            self.data[split] = {
                'df': pd.read_csv(csv_file),
                'path': split_path
            }
    
    def load_image(self, split, filename):
        img_path = self.data[split]['path'] / filename
        return cv2.imread(str(img_path))
    
    def get_label(self, split, filename):
        df = self.data[split]['df']
        return df[df['filename'] == filename]['text'].iloc[0]

import cv2
import numpy as np

import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    def preprocess(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Gentle denoising with non-local means
        denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
        denoised = cv2.fastNlMeansDenoising(denoised, None, h=20, templateWindowSize=5, searchWindowSize=15)
        denoised2 = cv2.fastNlMeansDenoising(denoised, None, h=15, templateWindowSize=5, searchWindowSize=10)
        # Text enhancement after joining
        gaussian_blurred = cv2.GaussianBlur(denoised2, (3, 3), 1.0)
        unsharp_masked = cv2.addWeighted(denoised2, 1.5, gaussian_blurred, -0.5, 0)
        # Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(unsharp_masked)
        
        # Edge-preserving smoothing
        text_enhanced = cv2.bilateralFilter(contrast_enhanced, d=5, sigmaColor=75, sigmaSpace=75)

        # Final adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            text_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
        )

        gaussian_blurred = cv2.GaussianBlur(adaptive, (3, 3), 1.0)

        adaptive = cv2.adaptiveThreshold(
            gaussian_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
        )

        return adaptive


def visualize_preprocessing_steps(images_dict):
    n_steps = len(images_dict)
    n_cols = 5
    n_rows = (n_steps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.ravel()
    for idx, (step_name, img) in enumerate(images_dict.items()):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(step_name)
        axes[idx].axis('off')
    # Hide empty subplots
    for idx in range(len(images_dict), len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    return fig

def main():
    # Initialize loader and preprocessor
    data_loader = DataLoader("./crnn_dataset")
    preprocessor = ImagePreprocessor()
    
    # Process 10 images from training set
    train_df = data_loader.data['train']['df']
    sample_images = train_df['filename'].head(10)
    
    # Create output directory for visualizations
    output_dir = Path("preprocessing_visualization")
    output_dir.mkdir(exist_ok=True)
    
    # Process and visualize each image
    for idx, filename in enumerate(sample_images):
        # Load and process image
        image = cv2.imread(str("/home/panesico/repositories/solbot/captcha2.jpg"))
        processed_images = preprocessor.preprocess(image)
        
        # Create visualization
        fig = visualize_preprocessing_steps(processed_images)
        
        # Save visualization
        fig.savefig(output_dir / f"preprocessing_steps_{idx}.png", 
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"Processed image {idx + 1}/10: {filename}")

if __name__ == "__main__":
    main()