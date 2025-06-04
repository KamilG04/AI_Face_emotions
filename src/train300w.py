import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split

class PTSLoader:
    """Loader for 300W dataset with .pts files"""
    
    def __init__(self, dataset_path: str, img_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            dataset_path: Path to the dataset directory
            img_size: Target size for resizing images (width, height)
        """
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.num_landmarks = 68  # Standard for 300W dataset
        self.landmark_connections = self._get_landmark_connections()
        
    def _get_landmark_connections(self) -> Dict[str, List[int]]:
        """Defines how landmarks should be connected for visualization"""
        return {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_bottom': list(range(31, 36)),
            'right_eye': list(range(36, 42)) + [36],  # Close the loop
            'left_eye': list(range(42, 48)) + [42],   # Close the loop
            'outer_lips': list(range(48, 60)) + [48],  # Close the loop
            'inner_lips': list(range(60, 68)) + [60]   # Close the loop
        }
    
    def parse_pts_file(self, pts_file_path: Path) -> np.ndarray:
        """Parse .pts file and return landmarks as numpy array
        
        Args:
            pts_file_path: Path to .pts file
            
        Returns:
            Numpy array of shape (68, 2) with landmark coordinates
        """
        with open(pts_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Find the start and end of landmark data
        try:
            start_idx = lines.index('{') + 1
            end_idx = lines.index('}')
        except ValueError:
            raise ValueError(f"Invalid .pts file format: {pts_file_path}")
        
        landmarks = []
        for line in lines[start_idx:end_idx]:
            if line:
                x, y = map(float, line.split())
                landmarks.append([x, y])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        if len(landmarks) != self.num_landmarks:
            raise ValueError(f"Expected {self.num_landmarks} landmarks, got {len(landmarks)}")
            
        return landmarks
    
    def load_image_and_landmarks(self, img_path: Path, pts_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and corresponding landmarks
        
        Args:
            img_path: Path to image file
            pts_path: Path to corresponding .pts file
            
        Returns:
            Tuple of (image, landmarks) where:
            - image: Resized and normalized image (0-1)
            - landmarks: Scaled landmarks matching the resized image
        """
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        # Load landmarks
        landmarks = self.parse_pts_file(pts_path)
        
        # Resize image and scale landmarks
        img_resized = cv2.resize(img, self.img_size)
        
        # Calculate scale factors
        scale_x = self.img_size[0] / original_w
        scale_y = self.img_size[1] / original_h
        
        # Scale landmarks
        landmarks_scaled = landmarks.copy()
        landmarks_scaled[:, 0] *= scale_x  # x coordinates
        landmarks_scaled[:, 1] *= scale_y  # y coordinates
        
        # Normalize image to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized, landmarks_scaled
    
    def scan_dataset(self) -> List[Tuple[Path, Path]]:
        """Scan dataset directory for all valid image+pts pairs
        
        Returns:
            List of tuples (image_path, pts_path)
        """
        pairs = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            for img_path in self.dataset_path.rglob(ext):
                pts_path = img_path.with_suffix('.pts')
                if pts_path.exists():
                    pairs.append((img_path, pts_path))
        
        if not pairs:
            raise FileNotFoundError(f"No valid image+pts pairs found in {self.dataset_path}")
            
        print(f"Found {len(pairs)} image-landmark pairs")
        return pairs
    
    def create_dataset(self, test_size: float = 0.2, val_size: float = 0.1, 
                      random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Create train/val/test splits
        
        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing tuples of (images, landmarks)
        """
        pairs = self.scan_dataset()
        
        # First split into train+val and test
        train_val_pairs, test_pairs = train_test_split(
            pairs, test_size=test_size, random_state=random_state)
        
        # Then split train_val into train and val
        train_pairs, val_pairs = train_test_split(
            train_val_pairs, test_size=val_size/(1-test_size), random_state=random_state)
        
        print(f"Dataset split:")
        print(f"- Train: {len(train_pairs)} samples")
        print(f"- Val: {len(val_pairs)} samples")
        print(f"- Test: {len(test_pairs)} samples")
        
        return {
            'train': self._load_pairs(train_pairs),
            'val': self._load_pairs(val_pairs),
            'test': self._load_pairs(test_pairs)
        }
    
    def _load_pairs(self, pairs: List[Tuple[Path, Path]]) -> Tuple[np.ndarray, np.ndarray]:
        """Load multiple image-landmark pairs
        
        Args:
            pairs: List of (image_path, pts_path) tuples
            
        Returns:
            Tuple of (images, landmarks) numpy arrays
        """
        images, landmarks = [], []
        
        for img_path, pts_path in pairs:
            try:
                img, lmk = self.load_image_and_landmarks(img_path, pts_path)
                images.append(img)
                landmarks.append(lmk)
            except Exception as e:
                print(f"Skipping {img_path}: {str(e)}")
                continue
        
        return np.array(images), np.array(landmarks)
    
    def visualize_landmarks(self, img: np.ndarray, landmarks: np.ndarray, 
                          title: str = "Face Landmarks", connections: bool = True):
        """Visualize landmarks on image
        
        Args:
            img: Image array (normalized 0-1)
            landmarks: Landmark coordinates
            title: Plot title
            connections: Whether to draw connections between landmarks
        """
        # Denormalize image if needed
        if img.max() <= 1.0:
            display_img = (img * 255).astype(np.uint8)
        else:
            display_img = img.astype(np.uint8)
            
        plt.figure(figsize=(10, 10))
        plt.imshow(display_img)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=10, label='Landmarks')
        
        if connections:
            for part, indices in self.landmark_connections.items():
                plt.plot(landmarks[indices, 0], landmarks[indices, 1], 
                        linewidth=2, alpha=0.7, label=part)
        
        plt.title(title)
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_random_samples(self, split: str = 'train', num_samples: int = 3):
        """Visualize random samples from a dataset split
        
        Args:
            split: Which split to visualize ('train', 'val', 'test')
            num_samples: Number of samples to display
        """
        pairs = []
        if split == 'train':
            pairs = self.scan_dataset()  # For demo purposes
        # In practice, you'd want to use your actual splits
        
        if not pairs:
            print(f"No pairs found for split {split}")
            return
            
        selected_pairs = np.random.choice(pairs, min(num_samples, len(pairs)), replace=False)
        
        for img_path, pts_path in selected_pairs:
            try:
                img, landmarks = self.load_image_and_landmarks(img_path, pts_path)
                self.visualize_landmarks(img, landmarks, title=img_path.name)
            except Exception as e:
                print(f"Error visualizing {img_path}: {e}")