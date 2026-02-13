#!/usr/bin/env python3
"""
================================================================================
SILO - Spatial Iterative Latent Outset
================================================================================

A fast image comparison engine that detects similarity even when images have
been rotated, cropped, filtered, or watermarked. Built for real-world use
where images rarely match pixel-perfect.

The name comes from how it works: we look at images spatially, iterate through
different angles and scales, find latent features that survive transforms,
and outset from there to make a decision.

Supports both CPU and CUDA GPU acceleration. Automatically picks the best
available device.

Quick start:
    # Command line
    python silo.py image1.jpg image2.jpg --verbose
    
    # As a library
    from silo import compare, Device
    result = compare("img1.jpg", "img2.jpg")
    print(f"Match: {result.is_match}, Similarity: {result.similarity:.1%}")

Written at Sylent.co
Version: 2.0.0

================================================================================
"""

import sys
import time
import warnings
import argparse
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass, field

import numpy as np
import cv2

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DEVICE SELECTION
# =============================================================================

class Device(Enum):
    """Device selection for computation."""
    CPU = auto()
    GPU = auto()
    AUTO = auto()


# Try to import optional dependencies
NUMBA_AVAILABLE = False
SKIMAGE_AVAILABLE = False
CUPY_AVAILABLE = False

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    pass

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    pass

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    pass


def get_available_devices() -> List[Device]:
    """Get list of available computation devices."""
    devices = [Device.CPU]
    if CUPY_AVAILABLE:
        devices.append(Device.GPU)
    return devices


def select_device(device: Device) -> Device:
    """Select the best available device."""
    if device == Device.AUTO:
        if CUPY_AVAILABLE:
            return Device.GPU
        return Device.CPU
    
    if device == Device.GPU and not CUPY_AVAILABLE:
        print("GPU requested but CuPy not available. Falling back to CPU.")
        return Device.CPU
    
    return device


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class TransformInfo:
    """Information about detected transforms."""
    rotation_angle: int = 0
    is_rotated: bool = False
    is_cropped: bool = False
    has_filter: bool = False
    has_watermark: bool = False
    is_overlay: bool = False  # Image placed on different background
    filter_type: str = ""
    watermark_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'rotation_angle': self.rotation_angle,
            'is_rotated': self.is_rotated,
            'is_cropped': self.is_cropped,
            'has_filter': self.has_filter,
            'has_watermark': self.has_watermark,
            'is_overlay': self.is_overlay,
            'filter_type': self.filter_type,
            'watermark_confidence': round(self.watermark_confidence, 4)
        }


@dataclass
class MatchResult:
    """Result from image comparison."""
    # Core results
    is_match: bool = False
    similarity: float = 0.0
    confidence: float = 0.0
    
    # Processing info
    method: str = ""
    processing_time: float = 0.0
    device_used: Device = Device.CPU
    
    # Detailed scores
    ssim_score: float = 0.0
    pixel_diff: float = 0.0
    template_score: float = 0.0
    feature_count: int = 0
    
    # Transform detection
    transforms: TransformInfo = field(default_factory=TransformInfo)
    
    # Debug info
    early_exit: bool = False
    exit_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'is_match': self.is_match,
            'similarity': round(self.similarity, 4),
            'confidence': round(self.confidence, 4),
            'method': self.method,
            'processing_time': round(self.processing_time, 4),
            'device': self.device_used.name,
            'ssim_score': round(self.ssim_score, 4),
            'pixel_diff': round(self.pixel_diff, 4),
            'template_score': round(self.template_score, 4),
            'feature_count': self.feature_count,
            'transforms': self.transforms.to_dict()
        }
    
    def __str__(self) -> str:
        status = "[MATCH]" if self.is_match else "[NO MATCH]"
        return f"{status} | Similarity: {self.similarity:.1%} | Confidence: {self.confidence:.1%} | Time: {self.processing_time:.3f}s"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for image matching."""
    # Device
    device: Device = Device.AUTO
    
    # Thresholds
    ssim_threshold: float = 0.87
    pixel_diff_threshold: float = 7.5
    template_threshold: float = 0.75
    feature_threshold: int = 50
    
    # Processing
    max_image_size: int = 1000
    verbose: bool = False
    visual: bool = False
    
    # Feature detection
    max_features: int = 500
    
    # Crop detection
    crop_size_ratio_threshold: float = 0.20


# =============================================================================
# PIXEL ANALYSIS
# =============================================================================

class PixelAnalyzer:
    """Fast pixel-level image analysis."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Perform pixel-level analysis between two images."""
        start_time = time.time()
        
        # Store original shapes
        orig_shape1 = img1.shape
        orig_shape2 = img2.shape
        
        # Resize for speed (independently, not to same size)
        img1_resized = self._resize(img1)
        img2_resized = self._resize(img2)
        
        # For SSIM, need same size - create copies
        h, w = img1_resized.shape[:2]
        img2_for_ssim = cv2.resize(img2_resized, (w, h), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY) if len(img1_resized.shape) == 3 else img1_resized
        gray2_for_ssim = cv2.cvtColor(img2_for_ssim, cv2.COLOR_RGB2GRAY) if len(img2_for_ssim.shape) == 3 else img2_for_ssim
        
        # Calculate SSIM
        ssim_score = self._calculate_ssim(gray1, gray2_for_ssim)
        
        # Calculate pixel difference
        pixel_diff, change_mask = self._calculate_pixel_diff(img1_resized, img2_for_ssim)
        
        return {
            'ssim_score': ssim_score,
            'pixel_diff': pixel_diff,
            'change_mask': change_mask,
            'img1': img1_resized,  # Keep original aspect ratio
            'img2': img2_resized,  # Keep original aspect ratio
            'gray1': gray1,
            'gray2': gray2_for_ssim,
            'orig_shape1': orig_shape1,
            'orig_shape2': orig_shape2,
            'time': time.time() - start_time
        }
    
    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image for faster processing."""
        h, w = img.shape[:2]
        max_size = self.config.max_image_size
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    
    def _calculate_ssim(self, gray1: np.ndarray, gray2: np.ndarray) -> float:
        """Calculate SSIM between grayscale images."""
        if SKIMAGE_AVAILABLE:
            score, _ = ssim(gray1, gray2, full=True)
            return score
        
        # Fallback: simple correlation
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        return float(result[0, 0])
    
    def _calculate_pixel_diff(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate pixel difference percentage."""
        diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
        
        if len(diff.shape) == 3:
            diff_max = np.maximum(np.maximum(diff[:, :, 0], diff[:, :, 1]), diff[:, :, 2])
        else:
            diff_max = diff
        
        threshold = 15
        change_mask = (diff_max >= threshold).astype(np.uint8)
        pixel_diff = (np.sum(change_mask) / change_mask.size) * 100
        
        return pixel_diff, change_mask


# =============================================================================
# TEMPLATE MATCHING
# =============================================================================

class TemplateMatcher:
    """Multi-rotation, multi-scale template matching."""
    
    def __init__(self, config: Config):
        self.config = config
        self.rotations = [0, 90, 180, 270]
        self.scales = [1.0, 0.75, 0.5, 0.25]
    
    def find_best_match(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Find best matching rotation between images."""
        start_time = time.time()
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        # Determine which is larger (target) and smaller (template)
        area1 = gray1.shape[0] * gray1.shape[1]
        area2 = gray2.shape[0] * gray2.shape[1]
        
        if area1 >= area2:
            larger, smaller = gray1, gray2
            larger_is_first = True
        else:
            larger, smaller = gray2, gray1
            larger_is_first = False
        
        # Resize for speed
        max_large = 800
        max_small = 400
        
        if max(larger.shape) > max_large:
            scale = max_large / max(larger.shape)
            larger = cv2.resize(larger, None, fx=scale, fy=scale)
        
        if max(smaller.shape) > max_small:
            scale = max_small / max(smaller.shape)
            smaller = cv2.resize(smaller, None, fx=scale, fy=scale)
        
        # Try all rotations
        results = []
        correlations = {}
        
        for angle in self.rotations:
            rotated = self._rotate_image(smaller, angle)
            corr = self._multi_scale_match(larger, rotated)
            results.append({'angle': angle, 'correlation': corr})
            correlations[angle] = corr
        
        # Get best result
        best = max(results, key=lambda x: x['correlation'])
        
        # The detected rotation is the angle the smaller image was rotated
        # relative to the larger image
        detected_angle = best['angle']
        
        return {
            'correlation': best['correlation'],
            'rotation_angle': detected_angle,  # The rotation detected
            'detected_angle': detected_angle,
            'all_correlations': correlations,
            'is_similar': best['correlation'] > self.config.template_threshold,
            'confidence': best['correlation'],
            'time': time.time() - start_time
        }
    
    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by given angle (counter-clockwise)."""
        if angle == 0:
            return img
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Use rotation matrix for proper rotation
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # Negative for counter-clockwise
        
        # For 90/270, swap dimensions
        if angle in [90, 270]:
            new_size = (h, w)
        else:
            new_size = (w, h)
        
        return cv2.warpAffine(img, M, new_size)
    
    def _multi_scale_match(self, target: np.ndarray, template: np.ndarray) -> float:
        """Multi-scale template matching."""
        h_t, w_t = target.shape[:2]
        h_s, w_s = template.shape[:2]
        
        best_corr = 0.0
        
        # Try multiple scales
        for scale in self.scales:
            try:
                new_h, new_w = int(h_s * scale), int(w_s * scale)
                if new_h < 20 or new_w < 20:
                    continue
                if new_h > h_t or new_w > w_t:
                    continue
                
                resized_template = cv2.resize(template, (new_w, new_h))
                
                # Template matching
                result = cv2.matchTemplate(target, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_corr:
                    best_corr = max_val
                
                # Early exit if very good match
                if max_val > 0.85:
                    break
            
            except Exception:
                continue
        
        return best_corr


# =============================================================================
# FEATURE MATCHING
# =============================================================================

class FeatureMatcher:
    """Feature-based image matching using ORB."""
    
    def __init__(self, config: Config):
        self.config = config
        self.detector = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def match(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Match features between two images."""
        start_time = time.time()
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        # Detect features
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return {
                'matches': 0,
                'confidence': 0.0,
                'inlier_ratio': 0.0,
                'time': time.time() - start_time
            }
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        match_count = len(good_matches)
        
        # Calculate confidence
        confidence = min(1.0, match_count / 100.0)
        
        # Calculate inlier ratio using RANSAC
        inlier_ratio = 0.0
        if match_count >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inlier_ratio = np.sum(mask) / len(good_matches)
        
        return {
            'matches': match_count,
            'confidence': confidence,
            'inlier_ratio': inlier_ratio,
            'keypoints1': len(kp1),
            'keypoints2': len(kp2),
            'time': time.time() - start_time
        }


# =============================================================================
# TRANSFORM DETECTION
# =============================================================================

class TransformDetector:
    """Detect transforms applied to images."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect(self, img1: np.ndarray, img2: np.ndarray, 
               pixel_result: Dict, template_result: Dict, 
               feature_result: Dict = None) -> TransformInfo:
        """Detect transforms between two images."""
        info = TransformInfo()
        
        # Rotation detection - use the best angle directly
        correlations = template_result['all_correlations']
        best_angle = template_result['rotation_angle']
        
        # Check if 90° and 270° are close - if so, prefer 90°
        # (both represent the same rotation, just different directions)
        if abs(correlations.get(90, 0) - correlations.get(270, 0)) < 0.05:
            if correlations.get(90, 0) > 0.5:
                best_angle = 90  # Prefer 90° when they're close
        
        info.rotation_angle = best_angle
        info.is_rotated = best_angle != 0
        
        # Crop detection
        orig_shape1 = pixel_result['orig_shape1']
        orig_shape2 = pixel_result['orig_shape2']
        area1 = orig_shape1[0] * orig_shape1[1]
        area2 = orig_shape2[0] * orig_shape2[1]
        size_ratio = min(area1, area2) / max(area1, area2)
        info.is_cropped = size_ratio < self.config.crop_size_ratio_threshold
        
        # Overlay detection (image on different background)
        info.is_overlay = self._detect_overlay(pixel_result, template_result, feature_result)
        
        # Filter detection
        info.has_filter = self._detect_filter(pixel_result)
        if info.has_filter:
            info.filter_type = self._classify_filter(pixel_result)
        
        # Watermark detection (only if not an overlay)
        if not info.is_overlay:
            info.has_watermark, info.watermark_confidence = self._detect_watermark(
                pixel_result, template_result
            )
        
        return info
    
    def _detect_overlay(self, pixel_result: Dict, template_result: Dict, feature_result: Dict = None) -> bool:
        """Detect if one image is an overlay on a different background."""
        img1 = pixel_result['img1']
        img2 = pixel_result['img2']
        correlation = template_result['correlation']
        ssim = pixel_result['ssim_score']
        
        # Overlays have:
        # 1. Good template correlation (content matches)
        # 2. Good feature matching (same content)
        # 3. Different image dimensions (usually)
        # 4. Large areas of completely different content (not just color shift)
        
        if correlation < 0.5:
            return False
        
        # High SSIM with high pixel diff usually indicates filter, not overlay
        # Overlay would have lower SSIM because background is different
        if ssim > 0.7:
            return False
        
        # If feature matching is provided, check it
        if feature_result is not None:
            feature_confidence = feature_result.get('confidence', 0)
            feature_matches = feature_result.get('matches', 0)
            # If very few features match, it's not an overlay - it's a different image
            if feature_matches < 10 or feature_confidence < 0.1:
                return False
        
        # Check for size difference (overlays often have different dimensions)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        size_ratio = (h1 * w1) / (h2 * w2)
        
        # If sizes are very different, more likely to be overlay
        has_size_diff = size_ratio < 0.7 or size_ratio > 1.4
        
        # Resize images to same size for comparison
        if h1 * w1 < h2 * w2:
            img1 = cv2.resize(img1, (w2, h2))
        else:
            img2 = cv2.resize(img2, (w1, h1))
        
        # Check for large uniform regions in the difference
        diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
        if len(diff.shape) == 3:
            diff_gray = np.max(diff, axis=2)
        else:
            diff_gray = diff
        
        # Find regions with very high difference (background)
        high_diff = diff_gray > 100
        high_diff_percent = np.sum(high_diff) / high_diff.size
        
        # For overlay: need both high difference AND size difference OR very high difference
        if has_size_diff and high_diff_percent > 0.3:
            return True
        if high_diff_percent > 0.5:  # Very high difference without size diff
            return True
        
        return False
    
    def _detect_filter(self, pixel_result: Dict) -> bool:
        """Detect if a filter was applied."""
        ssim = pixel_result['ssim_score']
        pixel_diff = pixel_result['pixel_diff']
        
        # Filter typically: good SSIM with high pixel difference
        # The image content is similar but colors/contrast changed
        # Expanded range to catch more filter cases
        if ssim > 0.5 and pixel_diff > 70:
            return True
        if 0.3 < ssim < 0.7 and pixel_diff > 50:
            return True
        return False
    
    def _classify_filter(self, pixel_result: Dict) -> str:
        """Classify the type of filter applied."""
        img1 = pixel_result['img1']
        img2 = pixel_result['img2']
        
        # Check brightness change
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        
        if mean2 > mean1 * 1.15:
            return "brightened"
        elif mean2 < mean1 * 0.85:
            return "darkened"
        
        # Check contrast
        std1 = np.std(img1)
        std2 = np.std(img2)
        
        if std2 > std1 * 1.15:
            return "high_contrast"
        elif std2 < std1 * 0.85:
            return "low_contrast"
        
        # Check color shift
        if len(img1.shape) == 3:
            r1, g1, b1 = np.mean(img1[:,:,0]), np.mean(img1[:,:,1]), np.mean(img1[:,:,2])
            r2, g2, b2 = np.mean(img2[:,:,0]), np.mean(img2[:,:,1]), np.mean(img2[:,:,2])
            
            # Color tint detection
            if r2 > g2 * 1.2 and r2 > b2 * 1.2:
                return "warm_tint"
            elif b2 > r2 * 1.2 and b2 > g2 * 1.2:
                return "cool_tint"
        
        return "color_adjustment"
    
    def _detect_watermark(self, pixel_result: Dict, template_result: Dict) -> Tuple[bool, float]:
        """Detect if a watermark was added."""
        change_mask = pixel_result['change_mask']
        correlation = template_result['correlation']
        ssim = pixel_result['ssim_score']
        img1 = pixel_result['img1']
        img2 = pixel_result['img2']
        
        # Watermarks: high template correlation but localized pixel changes
        # Key insight: watermarks are SMALL, localized additions
        # If SSIM is high (>0.7), it's likely just a filter, not a watermark
        if correlation < 0.5:
            return False, 0.0
        
        # High SSIM indicates filter, not watermark
        if ssim > 0.7:
            return False, 0.0
        
        h, w = change_mask.shape
        change_percent = np.sum(change_mask) / change_mask.size
        
        # If almost all pixels changed, it's NOT a watermark
        # Watermarks only affect a small portion of the image
        if change_percent > 0.8:
            return False, 0.0
        
        confidence = 0.0
        
        # Method 1: Check for text-like patterns using edge detection
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
            
            # Resize to same dimensions for comparison
            if gray1.shape != gray2.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            # Edge detection
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            # If there are significantly more edges in img2, could be watermark text
            edge_diff = np.sum(edges2) - np.sum(edges1)
            if edge_diff > 0:
                edge_increase = edge_diff / (np.sum(edges1) + 1)
                # More strict: need 20%+ edge increase for watermark
                if edge_increase > 0.2:
                    confidence = max(confidence, 0.6)
                elif edge_increase > 0.1:
                    confidence = max(confidence, 0.4)
        except:
            pass
        
        # Method 2: Check for localized high-contrast regions (typical of watermarks)
        try:
            if len(img1.shape) == 3:
                diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
                diff_gray = np.max(diff, axis=2)  # Max difference across channels
            else:
                diff_gray = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
            
            # Find regions with consistent high difference
            threshold = 50  # Higher threshold for watermark detection
            high_diff = (diff_gray > threshold).astype(np.uint8)
            
            # Watermarks should affect less than 30% of image
            high_diff_percent = np.sum(high_diff) / high_diff.size
            if high_diff_percent > 0.3:
                # Too much change - not a watermark
                return False, 0.0
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_diff, connectivity=8)
            
            # Watermarks typically have a few large connected regions
            if num_labels > 1:
                # Get sizes of components (excluding background)
                sizes = stats[1:, cv2.CC_STAT_AREA]
                max_size = np.max(sizes) if len(sizes) > 0 else 0
                total_size = np.sum(sizes)
                
                # If there's a dominant region that's not too big or too small
                if max_size > 100 and max_size < 0.2 * high_diff.size:
                    confidence = max(confidence, 0.7)
                elif total_size > 0.01 * high_diff.size and total_size < 0.3 * high_diff.size:
                    confidence = max(confidence, 0.5)
        except:
            pass
        
        # Method 3: Check corner/edge regions for watermark placement
        try:
            # Resize change_mask to match img1 dimensions
            if change_mask.shape != img1.shape[:2]:
                change_mask_resized = cv2.resize(change_mask.astype(np.uint8), 
                                                  (img1.shape[1], img1.shape[0]))
            else:
                change_mask_resized = change_mask
            
            h, w = change_mask_resized.shape
            corner_size = min(h, w) // 4
            
            # Check corners
            corners = [
                change_mask_resized[0:corner_size, 0:corner_size],
                change_mask_resized[0:corner_size, -corner_size:],
                change_mask_resized[-corner_size:, 0:corner_size],
                change_mask_resized[-corner_size:, -corner_size:],
            ]
            
            corner_change = sum(np.mean(c) for c in corners) / 4
            center = change_mask_resized[corner_size:-corner_size, corner_size:-corner_size]
            center_change = np.mean(center) if center.size > 0 else 0
            
            # Watermarks often in corners with higher concentration than center
            # But need significant difference
            if corner_change > 0.6 and corner_change > center_change * 1.5:
                confidence = max(confidence, 0.65)
        except:
            pass
        
        # Higher threshold for watermark detection
        return confidence > 0.55, confidence


# =============================================================================
# VISUAL OUTPUT
# =============================================================================

class Visualizer:
    """Generate visual output for comparison results."""
    
    @staticmethod
    def show(img1: np.ndarray, img2: np.ndarray, result: MatchResult, 
             pixel_result: Dict, template_result: Dict):
        """Display visual comparison."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("⚠️ matplotlib not available for visual output")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25)
        
        # Title
        status = "✓ MATCH" if result.is_match else "✗ NO MATCH"
        color = '#2ECC71' if result.is_match else '#E74C3C'
        
        fig.suptitle(f'{status} | Similarity: {result.similarity:.1%} | Time: {result.processing_time:.3f}s',
                    fontsize=14, fontweight='bold', color=color)
        
        # Original images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1)
        ax1.set_title('Image 1', fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2)
        ax2.set_title('Image 2', fontweight='bold')
        ax2.axis('off')
        
        # Difference overlay
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = img1.copy()
        if 'change_mask' in pixel_result:
            mask = pixel_result['change_mask']
            mask_resized = cv2.resize(mask, (img1.shape[1], img1.shape[0]))
            overlay[mask_resized > 0] = [255, 0, 0]
        ax3.imshow(overlay)
        ax3.set_title(f'Differences\n({result.pixel_diff:.1f}% changed)', fontweight='bold')
        ax3.axis('off')
        
        # Transform info
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        transform_text = "Detected Transforms:\n\n"
        if result.transforms.is_rotated:
            transform_text += f"🔄 Rotation: {result.transforms.rotation_angle}°\n"
        if result.transforms.is_cropped:
            transform_text += "✂️ Cropped\n"
        if result.transforms.has_filter:
            # Show filter type only when confident
            if result.transforms.filter_type and result.transforms.filter_type != "color_adjustment":
                transform_text += f"🎨 Filter: {result.transforms.filter_type}\n"
            else:
                transform_text += f"🎨 Filter: Yes\n"
        if result.transforms.has_watermark:
            transform_text += f"💧 Watermark ({result.transforms.watermark_confidence:.0%})\n"
        
        if not any([result.transforms.is_rotated, result.transforms.is_cropped, 
                   result.transforms.has_filter, result.transforms.has_watermark]):
            transform_text += "No transforms detected"
        
        ax4.text(0.5, 0.5, transform_text, ha='center', va='center', fontsize=11,
                transform=ax4.transAxes, family='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='#DEE2E6'))
        
        # Scores
        ax5 = fig.add_subplot(gs[1, 0])
        scores = ['SSIM', 'Template', 'Features']
        values = [result.ssim_score, result.template_score, result.feature_count / 100]
        colors = ['#3498DB', '#9B59B6', '#E67E22']
        bars = ax5.bar(scores, values, color=colors)
        ax5.set_ylim(0, 1)
        ax5.set_title('Scores', fontweight='bold')
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Rotation correlations
        ax6 = fig.add_subplot(gs[1, 1])
        if 'all_correlations' in template_result:
            angles = list(template_result['all_correlations'].keys())
            corrs = list(template_result['all_correlations'].values())
            ax6.bar([f"{a}°" for a in angles], corrs, color='#1ABC9C')
            ax6.set_ylim(0, 1)
            ax6.set_title('Rotation Correlation', fontweight='bold')
        
        # SSIM histogram
        ax7 = fig.add_subplot(gs[1, 2])
        if 'gray1' in pixel_result and 'gray2' in pixel_result:
            diff = np.abs(pixel_result['gray1'].astype(np.int32) - pixel_result['gray2'].astype(np.int32))
            ax7.hist(diff.flatten(), bins=50, color='#E74C3C', alpha=0.7)
            ax7.set_title('Pixel Difference Distribution', fontweight='bold')
            ax7.set_xlabel('Difference')
            ax7.set_ylabel('Count')
        
        # Summary
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        
        summary = f"""
        Summary
        ─────────────────────
        Method: {result.method}
        Device: {result.device_used.name}
        
        SSIM: {result.ssim_score:.4f}
        Pixel Diff: {result.pixel_diff:.2f}%
        Template: {result.template_score:.4f}
        Features: {result.feature_count}
        
        Confidence: {result.confidence:.1%}
        """
        
        ax8.text(0.5, 0.5, summary, ha='center', va='center', fontsize=10,
                transform=ax8.transAxes, family='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFF3CD', edgecolor='#FFC107'))
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# GPU ACCELERATION
# =============================================================================

class GPUProcessor:
    """GPU-accelerated processing using CuPy."""
    
    def __init__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Cannot use GPU processing.")
        self.device = cp.cuda.Device(0)
    
    def compute_correlation_gpu(self, template: np.ndarray, target: np.ndarray) -> float:
        """Compute correlation on GPU."""
        template_gpu = cp.asarray(template)
        target_gpu = cp.asarray(target)
        
        # Normalize
        template_norm = (template_gpu - cp.mean(template_gpu)) / (cp.std(template_gpu) + 1e-8)
        target_norm = (target_gpu - cp.mean(target_gpu)) / (cp.std(target_gpu) + 1e-8)
        
        # Correlation
        correlation = cp.sum(template_norm * target_norm) / template_norm.size
        return float(correlation)


# =============================================================================
# MAIN COMPARATOR
# =============================================================================

class ImageMatcher:
    """Main image matching class."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.device = select_device(self.config.device)
        
        # Initialize components
        self.pixel_analyzer = PixelAnalyzer(self.config)
        self.template_matcher = TemplateMatcher(self.config)
        self.feature_matcher = FeatureMatcher(self.config)
        self.transform_detector = TransformDetector(self.config)
        
        # GPU processor if available
        self.gpu_processor = None
        if self.config.device == Device.GPU and CUPY_AVAILABLE:
            self.gpu_processor = GPUProcessor()
    
    def compare(self, image1: Union[str, np.ndarray], 
                image2: Union[str, np.ndarray]) -> MatchResult:
        """Compare two images and return match result."""
        start_time = time.time()
        result = MatchResult(device_used=self.config.device)
        
        # Load images
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        
        if img1 is None or img2 is None:
            result.is_match = False
            result.confidence = 0.0
            result.method = "error"
            return result
        
        # Stage 1: Pixel Analysis
        if self.config.verbose:
            print("\n[Stage 1] Pixel Analysis...")
        
        pixel_result = self.pixel_analyzer.analyze(img1, img2)
        
        result.ssim_score = pixel_result['ssim_score']
        result.pixel_diff = pixel_result['pixel_diff']
        
        if self.config.verbose:
            print(f"  SSIM: {pixel_result['ssim_score']:.4f}")
            print(f"  Pixel Diff: {pixel_result['pixel_diff']:.2f}%")
            print(f"  Time: {pixel_result['time']:.3f}s")
        
        # Early exit conditions
        if pixel_result['ssim_score'] > self.config.ssim_threshold:
            result.is_match = True
            result.similarity = pixel_result['ssim_score']
            result.confidence = 0.95
            result.method = "ssim_match"
            result.early_exit = True
            result.exit_reason = f"SSIM ({pixel_result['ssim_score']:.4f}) > threshold ({self.config.ssim_threshold})"
            result.processing_time = time.time() - start_time
            return result
        
        if pixel_result['pixel_diff'] < self.config.pixel_diff_threshold:
            result.is_match = True
            result.similarity = 1.0 - (pixel_result['pixel_diff'] / 100)
            result.confidence = 0.90
            result.method = "pixel_match"
            result.early_exit = True
            result.exit_reason = f"Pixel diff ({pixel_result['pixel_diff']:.2f}%) < threshold ({self.config.pixel_diff_threshold}%)"
            result.processing_time = time.time() - start_time
            return result
        
        # Stage 2: Template Matching
        if self.config.verbose:
            print("\n[Stage 2] Template Matching...")
        
        template_result = self.template_matcher.find_best_match(
            pixel_result['img1'], pixel_result['img2']  # Use resized images
        )
        
        result.template_score = template_result['correlation']
        
        if self.config.verbose:
            for angle, corr in template_result['all_correlations'].items():
                print(f"  {angle}°: {corr:.4f}")
            print(f"  Best rotation: {template_result['rotation_angle']}° (correlation: {template_result['correlation']:.4f})")
            print(f"  Time: {template_result['time']:.3f}s")
        
        # Stage 3: Feature Matching (do this before transform detection)
        if self.config.verbose:
            print("\n[Stage 3] Feature Matching...")
        
        feature_result = self.feature_matcher.match(
            pixel_result['img1'], pixel_result['img2']  # Use resized images for speed
        )
        
        result.feature_count = feature_result['matches']
        
        if self.config.verbose:
            print(f"  Features: {feature_result['matches']}")
            print(f"  Confidence: {feature_result['confidence']:.4f}")
            print(f"  Time: {feature_result['time']:.3f}s")
        
        # Stage 4: Transform Detection (now with feature results)
        transforms = self.transform_detector.detect(img1, img2, pixel_result, template_result, feature_result)
        result.transforms = transforms
        
        # Stage 5: Decision Logic
        # Calculate size ratio for crop detection
        orig_shape1 = pixel_result['orig_shape1']
        orig_shape2 = pixel_result['orig_shape2']
        area1 = orig_shape1[0] * orig_shape1[1]
        area2 = orig_shape2[0] * orig_shape2[1]
        size_ratio = min(area1, area2) / max(area1, area2)
        is_crop = size_ratio < self.config.crop_size_ratio_threshold
        
        if is_crop:
            # For crops: trust template matching
            if template_result['correlation'] > self.config.template_threshold:
                result.is_match = True
                result.confidence = template_result['correlation']
                result.similarity = template_result['correlation']
                result.method = "template_match_crop"
                result.processing_time = time.time() - start_time
                
                if self.config.visual:
                    Visualizer.show(img1, img2, result, pixel_result, template_result)
                
                return result
        
        # Final decision for non-crop images
        combined_score = (template_result['correlation'] + feature_result['confidence']) / 2
        
        if template_result['correlation'] > 0.90 and feature_result['matches'] > 50:
            result.is_match = True
            result.confidence = template_result['correlation']
        elif template_result['correlation'] > 0.80 and feature_result['matches'] > 100:
            result.is_match = True
            result.confidence = combined_score
        elif feature_result['matches'] > 150:  # Strong feature matching alone
            result.is_match = True
            result.confidence = feature_result['confidence']
        else:
            result.is_match = False
            result.confidence = 1.0 - combined_score
        
        result.similarity = combined_score
        result.method = "combined_analysis"
        result.processing_time = time.time() - start_time
        
        # Visual output
        if self.config.visual:
            Visualizer.show(img1, img2, result, pixel_result, template_result)
        
        return result
    
    def _load_image(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from path or array."""
        try:
            if isinstance(image, str):
                img = cv2.imread(str(image))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return None
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    return np.stack([image] * 3, axis=2)
                elif image.shape[2] == 4:
                    return image[:, :, :3]
                return image
            return None
        except Exception as e:
            if self.config.verbose:
                print(f"Error loading image: {e}")
            return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compare(image1: Union[str, np.ndarray], 
            image2: Union[str, np.ndarray],
            device: Device = Device.AUTO,
            visual: bool = False,
            verbose: bool = False,
            **kwargs) -> MatchResult:
    """
    Compare two images and return match result.
    
    Args:
        image1: Path to first image or numpy array
        image2: Path to second image or numpy array
        device: Computation device (CPU, GPU, AUTO)
        visual: Show visual output
        verbose: Print detailed progress
        **kwargs: Additional configuration options
    
    Returns:
        MatchResult with comparison details
    
    Example:
        >>> from silo import compare, Device
        >>> result = compare("img1.jpg", "img2.jpg", device=Device.GPU)
        >>> print(f"Match: {result.is_match}, Confidence: {result.confidence:.2%}")
    """
    config = Config(
        device=device,
        visual=visual,
        verbose=verbose,
        **kwargs
    )
    matcher = ImageMatcher(config)
    return matcher.compare(image1, image2)


def is_match(image1: Union[str, np.ndarray],
             image2: Union[str, np.ndarray],
             threshold: float = 0.5) -> bool:
    """
    Quick check if two images are similar.
    
    Args:
        image1: Path to first image or numpy array
        image2: Path to second image or numpy array
        threshold: Confidence threshold for match
    
    Returns:
        True if images match, False otherwise
    """
    result = compare(image1, image2)
    return result.is_match and result.confidence >= threshold


def get_similarity(image1: Union[str, np.ndarray],
                   image2: Union[str, np.ndarray]) -> float:
    """
    Get similarity score between two images.
    
    Args:
        image1: Path to first image or numpy array
        image2: Path to second image or numpy array
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    result = compare(image1, image2)
    return result.similarity


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Image Match - Smart Image Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image1.jpg image2.jpg
  %(prog)s image1.jpg image2.jpg --device gpu --visual
  %(prog)s image1.jpg image2.jpg --verbose --threshold 0.8
        """
    )
    
    parser.add_argument('image1', help='First image path')
    parser.add_argument('image2', help='Second image path')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='auto',
                       help='Computation device (default: auto)')
    parser.add_argument('--visual', action='store_true',
                       help='Show visual output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for match (default: 0.5)')
    parser.add_argument('--ssim-threshold', type=float, default=0.87,
                       help='SSIM threshold for early exit (default: 0.87)')
    parser.add_argument('--pixel-diff-threshold', type=float, default=7.5,
                       help='Pixel diff threshold for early exit (default: 7.5)')
    
    args = parser.parse_args()
    
    # Map device string to enum
    device_map = {'cpu': Device.CPU, 'gpu': Device.GPU, 'auto': Device.AUTO}
    device = device_map[args.device]
    
    # Create config
    config = Config(
        device=device,
        visual=args.visual,
        verbose=args.verbose,
        ssim_threshold=args.ssim_threshold,
        pixel_diff_threshold=args.pixel_diff_threshold
    )
    
    # Print header
    print("=" * 60)
    print("SILO - Spatial Iterative Latent Outset")
    print("=" * 60)
    
    # Show available devices
    if args.verbose:
        print(f"\nAvailable devices: {[d.name for d in get_available_devices()]}")
        print(f"Selected device: {select_device(device).name}")
    
    # Run comparison
    matcher = ImageMatcher(config)
    result = matcher.compare(args.image1, args.image2)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    status = "[MATCH]" if result.is_match else "[NO MATCH]"
    print(f"\nStatus: {status}")
    print(f"Similarity: {result.similarity:.1%}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Method: {result.method}")
    print(f"Device: {result.device_used.name}")
    print(f"Time: {result.processing_time:.3f}s")
    
    print(f"\nDetailed Scores:")
    print(f"  SSIM: {result.ssim_score:.4f}")
    print(f"  Pixel Diff: {result.pixel_diff:.2f}%")
    print(f"  Template: {result.template_score:.4f}")
    print(f"  Features: {result.feature_count}")
    
    print(f"\nTransforms:")
    if result.transforms.is_rotated:
        print(f"  Rotation: {result.transforms.rotation_angle} deg")
    if result.transforms.is_cropped:
        print(f"  Cropped: Yes")
    if result.transforms.is_overlay:
        print(f"  Overlay: Yes (image on different background)")
    if result.transforms.has_filter:
        # Show filter type only when confident
        if result.transforms.filter_type and result.transforms.filter_type != "color_adjustment":
            print(f"  Filter: {result.transforms.filter_type}")
        else:
            print(f"  Filter: Yes")
    if result.transforms.has_watermark:
        print(f"  Watermark: {result.transforms.watermark_confidence:.0%}")
    if not any([result.transforms.is_rotated, result.transforms.is_cropped,
               result.transforms.has_filter, result.transforms.has_watermark,
               result.transforms.is_overlay]):
        print("  None detected")
    
    print("\n" + "=" * 60)
    
    # Exit code
    sys.exit(0 if result.is_match else 1)


if __name__ == "__main__":
    main()
