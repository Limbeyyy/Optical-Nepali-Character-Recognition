import os
import cv2
import torch
from torchvision import transforms
from typing import List, Union

class InferencePreprocessor:
    """
    Preprocess images for inference with EnhancedBMCNNwHFCs
    """

    def __init__(self, img_size=(32, 32), device=None):
        """
        Args:
            img_size: Resize all images to (H, W) expected by model
            device: torch.device('cuda') or torch.device('cpu')
        """
        self.img_size = img_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Transform pipeline (match training transforms)
        self.transform = transforms.Compose([
            transforms.ToTensor(),                # HxWxC -> CxHxW
            transforms.Normalize((0.5,), (0.5,)) # match training normalization
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load a single image and preprocess for inference
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Resize to model input
        img = cv2.resize(img, self.img_size)
        
        # Apply transform
        tensor = self.transform(img)  # shape: [1,H,W]
        tensor = tensor.unsqueeze(0)  # add batch dimension -> [1,1,H,W]
        tensor = tensor.to(self.device)
        return tensor

    def preprocess_folder(self, folder_path: str) -> List[torch.Tensor]:
        """
        Preprocess all images in a folder
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        images = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, filename)
                tensor = self.preprocess_image(img_path)
                images.append(tensor)
        return images

    def preprocess_paths(self, image_paths: List[str]) -> List[torch.Tensor]:
        """
        Preprocess a list of image paths
        """
        tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path)
            tensors.append(tensor)
        return tensors

