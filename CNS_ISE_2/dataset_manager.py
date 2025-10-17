"""
Dataset Manager & Comprehensive Evaluation Metrics
===================================================
Manages dataset preparation and tracks detailed performance metrics
including all evaluation criteria for lightweight validation.
"""

import os
import time
import json
import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import zipfile
from datetime import datetime


class ComprehensiveMetrics:
    """
    Tracks all performance and quality metrics for lightweight validation
    """
    
    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.metrics = {
            # Time metrics (milliseconds)
            't_total_encode': [],
            't_keygen': [],
            't_key_exchange': [],
            't_encrypt': [],
            't_mapping': [],
            't_embedding': [],
            't_io_write': [],
            't_total_decode': [],
            't_io_read': [],
            't_extract': [],
            't_decrypt': [],
            
            # Memory metrics (MB)
            'peak_memory_encode': [],
            'peak_memory_decode': [],
            'memory_baseline': [],
            
            # CPU metrics
            'cpu_core_seconds_encode': [],
            'cpu_core_seconds_decode': [],
            
            # Algorithmic metrics
            'modular_inversion_count': [],
            'ecc_point_operations': [],
            
            # Image quality metrics
            'psnr_db': [],
            'ssim': [],
            'mse': [],
            
            # Capacity metrics
            'payload_size_bytes': [],
            'message_size_bytes': [],
            'overhead_bytes': [],
            'bits_per_pixel': [],
            'embedding_efficiency': [],
            
            # Success metrics
            'message_integrity': [],
            'decode_success': []
        }
        
        self.test_count = 0
    
    def add_encode_metrics(self, **kwargs):
        """Add encoding metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        self.test_count += 1
    
    def add_decode_metrics(self, **kwargs):
        """Add decoding metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def add_quality_metrics(self, **kwargs):
        """Add image quality metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_statistics(self) -> Dict:
        """Calculate statistics for all metrics"""
        stats = {'phase': self.phase_name, 'test_count': self.test_count}
        
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                # Ensure numeric array for stats (bool -> float to avoid numpy boolean subtract issues)
                arr = np.asarray(values)
                if arr.dtype == np.bool_:
                    arr = arr.astype(float)
                stats[metric_name] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'p95': float(np.percentile(arr, 95)),
                    'p99': float(np.percentile(arr, 99))
                }
        
        return stats
    
    def export_to_json(self, filepath: str):
        """Export all metrics to JSON"""
        stats = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


class ImageQualityMetrics:
    """Calculate image quality metrics (PSNR, SSIM, MSE)"""
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, stego: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio
        Higher is better (>50dB is excellent for steganography)
        """
        mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, stego: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index
        Range: 0-1, higher is better (>0.99 is excellent)
        """
        # Simple SSIM implementation for RGB
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        original = original.astype(float)
        stego = stego.astype(float)
        
        mu1 = original.mean()
        mu2 = stego.mean()
        sigma1_sq = original.var()
        sigma2_sq = stego.var()
        sigma12 = np.mean((original - mu1) * (stego - mu2))
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim
    
    @staticmethod
    def calculate_mse(original: np.ndarray, stego: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return np.mean((original.astype(float) - stego.astype(float)) ** 2)


class DatasetManager:
    """
    Manages dataset preparation for benchmarking
    Supports both synthetic generation and real dataset download
    """
    
    def __init__(self, base_dir: str = "dataset"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.texts_dir = self.base_dir / "texts"
        self.metadata_file = self.base_dir / "metadata.json"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.texts_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_dataset(
        self, 
        num_images: int = 100,
        image_size: Tuple[int, int] = (512, 512),
        pattern_types: List[str] = None
    ):
        """
        Generate synthetic dataset for testing
        Use this for initial testing before getting real dataset
        """
        print(f"\n{'='*70}")
        print(f"GENERATING SYNTHETIC DATASET")
        print(f"{'='*70}")
        print(f"Images: {num_images}")
        print(f"Size: {image_size[0]}x{image_size[1]}")
        print(f"Location: {self.images_dir}")
        
        if pattern_types is None:
            pattern_types = ['gradient', 'noise', 'checkerboard', 'circles', 'waves']
        
        metadata = {
            'type': 'synthetic',
            'num_images': num_images,
            'image_size': image_size,
            'created': datetime.now().isoformat(),
            'images': []
        }
        
        for i in range(num_images):
            pattern_type = pattern_types[i % len(pattern_types)]
            img_array = self._generate_pattern_image(image_size, pattern_type, i)
            
            # Save image
            img_path = self.images_dir / f"synthetic_{i:04d}.png"
            Image.fromarray(img_array).save(img_path, 'PNG')
            
            # Add metadata
            metadata['images'].append({
                'filename': img_path.name,
                'pattern': pattern_type,
                'index': i,
                'hash': hashlib.sha256(img_array.tobytes()).hexdigest()[:16]
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{num_images} images...")
        
        # Generate corresponding text messages
        text_messages = self._generate_text_messages(num_images)
        metadata['texts'] = []
        
        for i, text in enumerate(text_messages):
            text_path = self.texts_dir / f"message_{i:04d}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            metadata['texts'].append({
                'filename': text_path.name,
                'length': len(text),
                'index': i
            })
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Generated {num_images} images and texts")
        print(f"✓ Metadata saved to {self.metadata_file}")
        print(f"✓ Total size: ~{self._estimate_dataset_size(num_images, image_size):.1f} MB")
        
        return metadata
    
    def _generate_pattern_image(
        self, 
        size: Tuple[int, int], 
        pattern_type: str, 
        seed: int
    ) -> np.ndarray:
        """Generate different pattern types for varied testing"""
        np.random.seed(seed)
        height, width = size
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if pattern_type == 'gradient':
            for i in range(height):
                img[i, :, :] = int(255 * i / height)
        
        elif pattern_type == 'noise':
            img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        elif pattern_type == 'checkerboard':
            block_size = 20
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    if (i // block_size + j // block_size) % 2 == 0:
                        img[i:i+block_size, j:j+block_size] = 255
        
        elif pattern_type == 'circles':
            center_y, center_x = height // 2, width // 2
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= (min(height, width) // 4)**2
            img[mask] = 255
        
        elif pattern_type == 'waves':
            for i in range(height):
                wave = 128 + 127 * np.sin(2 * np.pi * i / 50)
                img[i, :, :] = int(wave)
        
        return img
    
    def _generate_text_messages(self, count: int) -> List[str]:
        """Generate varied test messages"""
        templates = [
            "Short message {i}.",
            "This is a medium-length test message number {i} for cryptography research.",
            "A longer message {i} containing more data to test the capacity and performance of our steganography system more thoroughly.",
            "Special characters test {i}: @#$%^&*()!",
            "Numbers only: {i}234567890",
            "Mixed content {i}: Test123!@# with symbols and numbers",
            "Lorem ipsum dolor sit amet, consectetur adipiscing {i}.",
            "Cryptography ensures confidentiality and integrity {i}.",
            "Edge computing requires lightweight algorithms for IoT devices {i}.",
            "Steganography hides data in plain sight {i}."
        ]
        
        messages = []
        for i in range(count):
            template = templates[i % len(templates)]
            messages.append(template.format(i=i))
        
        return messages
    
    def download_real_dataset(self, dataset_name: str = "coco_subset"):
        """
        Download and prepare real image dataset
        Options: 'coco_subset', 'imagenet_subset', 'cifar10'
        """
        print(f"\n{'='*70}")
        print(f"DOWNLOADING REAL DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        # Note: This is a placeholder for real dataset download
        # You would implement actual download logic here
        
        print("⚠ Real dataset download not implemented in this template")
        print("Options:")
        print("  1. Download COCO dataset manually from https://cocodataset.org/")
        print("  2. Use ImageNet subset")
        print("  3. Use this synthetic generator for now")
        print("\nFor research paper validation:")
        print("  - Use at least 100-200 diverse images")
        print("  - Ensure images are 512x512 PNG")
        print("  - Include varied content (natural, synthetic, different complexities)")
    
    def prepare_standard_benchmark_set(
        self,
        num_images: int = 200,
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Prepare standardized benchmark set meeting research requirements
        """
        print(f"\n{'='*70}")
        print(f"PREPARING STANDARD BENCHMARK SET")
        print(f"{'='*70}")
        print(f"Target: {num_images} images at {image_size[0]}x{image_size[1]}")
        print(f"Estimated size: ~{self._estimate_dataset_size(num_images, image_size):.1f} MB")
        
        # Generate synthetic dataset
        metadata = self.generate_synthetic_dataset(num_images, image_size)
        
        # Validate dataset
        self._validate_dataset(metadata)
        
        return metadata
    
    def _validate_dataset(self, metadata: Dict):
        """Validate dataset meets requirements"""
        print(f"\n{'='*70}")
        print("DATASET VALIDATION")
        print(f"{'='*70}")
        
        num_images = len(metadata['images'])
        num_texts = len(metadata['texts'])
        
        checks = {
            'Image count': (num_images >= 100, f"{num_images} images"),
            'Text count': (num_texts >= 100, f"{num_texts} texts"),
            'Match': (num_images == num_texts, "Images match texts"),
            'Format': (all(img['filename'].endswith('.png') for img in metadata['images']), "All PNG"),
            'Size': (metadata['image_size'] == (512, 512), f"{metadata['image_size']}")
        }
        
        for check_name, (passed, detail) in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}: {detail}")
        
        all_passed = all(passed for passed, _ in checks.values())
        
        if all_passed:
            print(f"\n✓ Dataset validation PASSED")
        else:
            print(f"\n✗ Dataset validation FAILED - review requirements")
        
        return all_passed
    
    @staticmethod
    def _estimate_dataset_size(num_images: int, image_size: Tuple[int, int]) -> float:
        """Estimate dataset size in MB"""
        # PNG compression varies, estimate ~3 bytes per pixel
        bytes_per_image = image_size[0] * image_size[1] * 3 * 0.7  # 70% compression estimate
        total_mb = (bytes_per_image * num_images) / (1024 * 1024)
        return total_mb
    
    def load_dataset(self) -> Tuple[List[Path], List[str]]:
        """Load dataset for benchmarking"""
        if not self.metadata_file.exists():
            raise FileNotFoundError("Dataset not prepared. Run prepare_standard_benchmark_set() first")
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        image_paths = [self.images_dir / img['filename'] for img in metadata['images']]
        
        text_messages = []
        for text_info in metadata['texts']:
            text_path = self.texts_dir / text_info['filename']
            with open(text_path, 'r', encoding='utf-8') as f:
                text_messages.append(f.read())
        
        return image_paths, text_messages


def demonstrate_metrics_usage():
    """Demonstrate how to use comprehensive metrics"""
    print("\n" + "="*70)
    print("COMPREHENSIVE METRICS DEMONSTRATION")
    print("="*70)
    
    # Initialize metrics tracker
    metrics = ComprehensiveMetrics("Phase 2 Optimized")
    
    # Simulate encoding metrics
    print("\nSimulating encode operation...")
    metrics.add_encode_metrics(
        t_total_encode=85.3,
        t_keygen=12.5,
        t_key_exchange=8.2,
        t_encrypt=15.1,
        t_mapping=5.3,
        t_embedding=35.2,
        t_io_write=9.0,
        peak_memory_encode=42.5,
        cpu_core_seconds_encode=0.15,
        payload_size_bytes=150,
        message_size_bytes=100,
        overhead_bytes=50,
        bits_per_pixel=0.0019,
        embedding_efficiency=0.67
    )
    
    # Simulate decode metrics
    print("Simulating decode operation...")
    metrics.add_decode_metrics(
        t_total_decode=65.2,
        t_io_read=8.5,
        t_extract=30.1,
        t_decrypt=26.6,
        peak_memory_decode=38.2,
        message_integrity=True,
        decode_success=True
    )
    
    # Simulate quality metrics
    print("Simulating quality metrics...")
    metrics.add_quality_metrics(
        psnr_db=54.2,
        ssim=0.9987,
        mse=0.023
    )
    
    # Get statistics
    stats = metrics.get_statistics()
    
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    
    key_metrics = [
        ('t_total_encode', 'Total Encode Time', 'ms'),
        ('t_total_decode', 'Total Decode Time', 'ms'),
        ('peak_memory_encode', 'Peak Memory', 'MB'),
        ('psnr_db', 'PSNR', 'dB'),
        ('ssim', 'SSIM', 'ratio'),
        ('bits_per_pixel', 'Bits per Pixel', 'bpp')
    ]
    
    for metric_key, metric_name, unit in key_metrics:
        if metric_key in stats:
            value = stats[metric_key]['mean']
            print(f"{metric_name:.<30} {value:.3f} {unit}")
    
    # Export to JSON
    output_file = "comprehensive_metrics_demo.json"
    metrics.export_to_json(output_file)
    print(f"\n✓ Metrics exported to {output_file}")


if __name__ == "__main__":
    # Demonstrate dataset preparation
    dataset_mgr = DatasetManager("benchmark_dataset")
    
    print("\n" + "="*70)
    print("DATASET PREPARATION FOR RESEARCH PAPER")
    print("="*70)
    
    print("\nOptions:")
    print("1. Generate synthetic dataset (100 images) - Quick testing")
    print("2. Generate full benchmark set (200 images) - Research paper")
    print("3. Demonstrate metrics tracking")
    
    choice = input("\nSelect option [1-3]: ").strip() or "1"
    
    if choice == "1":
        dataset_mgr.generate_synthetic_dataset(num_images=100)
    elif choice == "2":
        dataset_mgr.prepare_standard_benchmark_set(num_images=200)
    elif choice == "3":
        demonstrate_metrics_usage()
    
    print("\n✓ Dataset preparation complete!")
    print("\nNext steps:")
    print("  1. Use this dataset for Phase 3 benchmarking")
    print("  2. Track all comprehensive metrics during tests")
    print("  3. Include metrics in research paper results section")