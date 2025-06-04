import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

class PTSLoader:
    """Loader dla 300W dataset z plikami .pts - poprawiony"""
    
    def __init__(self, dataset_path, img_size=(224, 224)):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.num_landmarks = 68  # Standard dla 300W
        
    def parse_pts_file(self, pts_file_path):
        """Parsuje plik .pts i zwraca landmarks jako numpy array"""
        landmarks = []
        
        with open(pts_file_path, 'r') as f:
            lines = f.readlines()
            
        # Pomijamy header i footer
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith('n_points'):
                continue
            elif line.strip() == '{':
                start_idx = i + 1
            elif line.strip() == '}':
                end_idx = i
                break
        
        if start_idx is None or end_idx is None:
            raise ValueError(f"Invalid .pts file format: {pts_file_path}")
        
        # Parsuj współrzędne
        for i in range(start_idx, end_idx):
            line = lines[i].strip()
            if line:
                x, y = map(float, line.split())
                landmarks.append([x, y])
        
        return np.array(landmarks, dtype=np.float32)
    
    def load_image_and_landmarks(self, img_path, pts_path):
        """Ładuje zdjęcie i odpowiadające mu landmarks z WŁAŚCIWĄ NORMALIZACJĄ"""
        # Ładuj zdjęcie
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]  # (height, width)
        
        # Ładuj landmarks
        landmarks = self.parse_pts_file(pts_path)
        
        # Resize image
        img_resized = cv2.resize(img, self.img_size)
        
        # Przeskaluj landmarks do nowego rozmiaru
        scale_x = self.img_size[0] / original_shape[1]  # width
        scale_y = self.img_size[1] / original_shape[0]  # height
        
        landmarks_scaled = landmarks.copy()
        landmarks_scaled[:, 0] *= scale_x  # x coordinates
        landmarks_scaled[:, 1] *= scale_y  # y coordinates
        
        # KLUCZOWE: Normalizuj landmarks do zakresu [-1, 1]
        # Przekształć z [0, 224] na [-1, 1]
        landmarks_normalized = (landmarks_scaled / 112.0) - 1.0
        
        # Normalizuj obraz do [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized, landmarks_normalized
    
    def scan_dataset(self):
        """Skanuje dataset i znajduje wszystkie pary img+pts"""
        pairs = []
        
        # Skanuj wszystkie foldery w dataset_path
        for img_ext in ['*.jpg', '*.png', '*.jpeg']:
            for img_path in self.dataset_path.rglob(img_ext):
                # Szukaj odpowiadającego pliku .pts
                pts_path = img_path.with_suffix('.pts')
                
                if pts_path.exists():
                    pairs.append((img_path, pts_path))
        
        print(f"Znaleziono {len(pairs)} par zdjęcie+landmarks")
        return pairs
    
    def create_dataset(self, test_split=0.2, val_split=0.1, random_state=42):
        """Tworzy dataset podzielony na train/val/test"""
        pairs = self.scan_dataset()
        
        if len(pairs) == 0:
            raise ValueError("Nie znaleziono żadnych par zdjęcie+landmarks!")
        
        # Najpierw podziel na train+val i test
        train_val_pairs, test_pairs = train_test_split(
            pairs, test_size=test_split, random_state=random_state
        )
        
        # Potem podziel train+val na train i val
        train_pairs, val_pairs = train_test_split(
            train_val_pairs, test_size=val_split/(1-test_split), random_state=random_state
        )
        
        print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
        
        return {
            'train': self.load_pairs(train_pairs),
            'val': self.load_pairs(val_pairs),
            'test': self.load_pairs(test_pairs)
        }
    
    def load_pairs(self, pairs):
        """Ładuje wszystkie pary i zwraca jako numpy arrays"""
        images = []
        landmarks = []
        
        for img_path, pts_path in pairs:
            try:
                img, lmks = self.load_image_and_landmarks(img_path, pts_path)
                images.append(img)
                landmarks.append(lmks)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        return np.array(images), np.array(landmarks)
    
    def create_tf_dataset(self, images, landmarks, batch_size=32, shuffle=True, augment=False):
        """Tworzy TensorFlow dataset z augmentacją"""
        dataset = tf.data.Dataset.from_tensor_slices((images, landmarks))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        if augment:
            dataset = dataset.map(self.augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def augment_data(self, image, landmarks):
        """Augmentacja danych - UWAGA: landmarks są w [-1, 1]"""
        # Random brightness
        image = tf.image.random_brightness(image, 0.2)
        
        # Random contrast
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random saturation
        image = tf.image.random_saturation(image, 0.7, 1.3)
        
        # Random horizontal flip z odpowiednim przekształceniem landmarks
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            # Flip landmarks (x-coordinate)
            landmarks = tf.stack([
                -landmarks[:, 0],  # Odbij x (bo są w [-1, 1])
                landmarks[:, 1]
            ], axis=1)
            
            # Trzeba też zmienić kolejność niektórych punktów (np. lewe/prawe oko)
            # To wymaga bardziej złożonej logiki - uproszczam
        
        # Ensure image is in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, landmarks
    
    def visualize_landmarks(self, img, landmarks, title="Face Landmarks"):
        """Wizualizuje landmarks na zdjęciu"""
        plt.figure(figsize=(8, 8))
        
        # Denormalizuj obraz jeśli trzeba
        if img.max() <= 1.0:
            display_img = (img * 255).astype(np.uint8)
        else:
            display_img = img.astype(np.uint8)
            
        plt.imshow(display_img)
        
        # Denormalizuj landmarks z [-1, 1] do [0, 223]
        landmarks_pixels = (landmarks + 1.0) * 112.0
        
        # Plot all landmarks
        plt.scatter(landmarks_pixels[:, 0], landmarks_pixels[:, 1], c='red', s=10)
        
        # Opcjonalnie: połącz punkty dla różnych części twarzy
        # Jaw line (0-16)
        jaw = landmarks_pixels[0:17]
        plt.plot(jaw[:, 0], jaw[:, 1], 'b-', linewidth=1, alpha=0.7)
        
        # Right eyebrow (17-21)
        r_brow = landmarks_pixels[17:22]
        plt.plot(r_brow[:, 0], r_brow[:, 1], 'g-', linewidth=1, alpha=0.7)
        
        # Left eyebrow (22-26)
        l_brow = landmarks_pixels[22:27]
        plt.plot(l_brow[:, 0], l_brow[:, 1], 'g-', linewidth=1, alpha=0.7)
        
        # Nose (27-35)
        nose = landmarks_pixels[27:36]
        plt.plot(nose[:, 0], nose[:, 1], 'm-', linewidth=1, alpha=0.7)
        
        # Right eye (36-41)
        r_eye = np.vstack([landmarks_pixels[36:42], landmarks_pixels[36]])
        plt.plot(r_eye[:, 0], r_eye[:, 1], 'c-', linewidth=1, alpha=0.7)
        
        # Left eye (42-47)
        l_eye = np.vstack([landmarks_pixels[42:48], landmarks_pixels[42]])
        plt.plot(l_eye[:, 0], l_eye[:, 1], 'c-', linewidth=1, alpha=0.7)
        
        # Mouth (48-67)
        outer_lip = np.vstack([landmarks_pixels[48:60], landmarks_pixels[48]])
        plt.plot(outer_lip[:, 0], outer_lip[:, 1], 'r-', linewidth=1, alpha=0.7)
        
        inner_lip = np.vstack([landmarks_pixels[60:68], landmarks_pixels[60]])
        plt.plot(inner_lip[:, 0], inner_lip[:, 1], 'r-', linewidth=1, alpha=0.7)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def denormalize_landmarks(self, landmarks):
        """Przekształca landmarks z [-1, 1] na [0, 223]"""
        return (landmarks + 1.0) * 112.0
    
    def normalize_landmarks(self, landmarks):
        """Przekształca landmarks z [0, 223] na [-1, 1]"""
        return (landmarks / 112.0) - 1.0