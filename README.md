# SILO - Spatial Iterative Latent Outset

A fast image comparison engine that detects similarity even when images have been rotated, cropped, filtered, or watermarked. Built for real-world use where images rarely match pixel-perfect.

**Written at [Sylent.co](https://sylent.co)**

---

## What's in the Name?

**SILO** stands for **Spatial Iterative Latent Outset**:

- **Spatial**: We analyze images by looking at their spatial structure, not just raw pixels
- **Iterative**: We try multiple angles (0°, 90°, 180°, 270°) and scales to find the best match
- **Latent**: We extract features that survive transformations - the "essence" of the image
- **Outset**: We start from these features and work outward to make a decision

---

## Installation

```bash
pip install -r requirements.txt
```

This installs the core dependencies. For GPU acceleration, uncomment the appropriate `cupy` line in `requirements.txt` based on your CUDA version.

---

## Quick Start

### Command Line

```bash
# Basic comparison
python silo.py image1.jpg image2.jpg

# With detailed output
python silo.py image1.jpg image2.jpg --verbose

# Use GPU acceleration
python silo.py image1.jpg image2.jpg --device gpu

# Show visual comparison
python silo.py image1.jpg image2.jpg --visual
```

### As a Library

```python
from silo import compare, Device

# Simple comparison
result = compare("image1.jpg", "image2.jpg")
print(f"Match: {result.is_match}")
print(f"Similarity: {result.similarity:.1%}")

# With GPU acceleration
result = compare("image1.jpg", "image2.jpg", device=Device.GPU)

# Check what transforms were detected
if result.transforms.is_rotated:
    print(f"Rotation: {result.transforms.rotation_angle}°")
if result.transforms.is_cropped:
    print("Image was cropped")
if result.transforms.has_filter:
    print("Filter was applied")
```

---

## How It Works

SILO uses a 4-stage pipeline to compare images:

### Stage 1: Pixel Analysis
We compute SSIM (Structural Similarity Index) and pixel-level differences. If images are nearly identical or completely different, we can exit early.

### Stage 2: Template Matching
We slide one image over the other at multiple rotations (0°, 90°, 180°, 270°) and scales. This catches cases where an image was rotated or is a cropped portion of another.

### Stage 3: Feature Matching
We detect ORB features (corners, edges, blobs) in both images and match them. If enough features align, the images share content. RANSAC filters out false matches.

### Stage 4: Transform Detection
We analyze the differences to detect what happened:
- **Rotation**: Which angle gives the best match?
- **Crop**: Is one image a subset of another?
- **Filter**: Was brightness/contrast/color adjusted?
- **Watermark**: Was text or a logo added?
- **Overlay**: Is the image placed on a different background?

---

## Test Results

Here's how SILO performs on real test cases:

### Test 1: Filtered Image
```
Images: bird.jpg vs bird2.jpg
Result: MATCH (77.6% similarity)
Transforms: Filter applied (color adjustment)
Time: 2.9s
```
The second image had its colors shifted, but SILO still recognized it as the same bird.

### Test 2: Rotated + Cropped
```
Images: img11.jpg vs img22.jpg
Result: MATCH (84.8% similarity)
Transforms: 90° rotation, cropped, warm tint filter
Time: 1.6s
```
The second image was rotated 90°, cropped to a smaller region, and had a warm filter applied.

### Test 3: Overlay on Different Background
```
Images: img11.jpg vs img33.jpg
Result: MATCH (86.5% similarity)
Transforms: 90° rotation, overlay on different background
Time: 0.8s
```
The content from the first image was placed on a completely different background.

### Test 4: Different Images
```
Images: img11.jpg vs gg.png
Result: NO MATCH (37.9% similarity)
Time: 0.8s
```
Completely different images are correctly identified as not matching.

---

## API Reference

### `compare(image1, image2, device=Device.AUTO, visual=False, verbose=False, **kwargs)`

Compare two images and return a result.

**Parameters:**
- `image1` (str or np.ndarray): First image path or array
- `image2` (str or np.ndarray): Second image path or array
- `device` (Device): Computation device - `Device.CPU`, `Device.GPU`, or `Device.AUTO`
- `visual` (bool): Show matplotlib visualization
- `verbose` (bool): Print detailed progress

**Returns:** `MatchResult` object

### `MatchResult`

| Field | Type | Description |
|-------|------|-------------|
| `is_match` | bool | True if images are similar |
| `similarity` | float | Similarity score (0.0 to 1.0) |
| `confidence` | float | Confidence in the result |
| `method` | str | How the decision was made |
| `processing_time` | float | Time in seconds |
| `ssim_score` | float | SSIM similarity |
| `template_score` | float | Template match correlation |
| `feature_count` | int | Number of matching features |
| `transforms` | TransformInfo | Detected transforms |

### `TransformInfo`

| Field | Type | Description |
|-------|------|-------------|
| `rotation_angle` | int | Detected rotation (0, 90, 180, 270) |
| `is_rotated` | bool | True if rotation detected |
| `is_cropped` | bool | True if one image is a crop |
| `is_overlay` | bool | True if image on different background |
| `has_filter` | bool | True if filter was applied |
| `filter_type` | str | Type of filter (if detected) |
| `has_watermark` | bool | True if watermark detected |

### `is_match(image1, image2, threshold=0.5)`

Quick boolean check if two images are similar.

```python
from silo import is_match

if is_match("img1.jpg", "img2.jpg"):
    print("Images are similar!")
```

---

## Configuration

You can fine-tune SILO by passing configuration options:

```python
from silo import ImageMatcher, Config, Device

config = Config(
    device=Device.GPU,
    ssim_threshold=0.95,        # SSIM above this = instant match
    pixel_diff_threshold=5.0,   # Pixel diff below this = instant match
    template_threshold=0.65,    # Template correlation threshold
    max_features=500,           # Max ORB features to detect
    max_image_size=1000,        # Resize large images for speed
    verbose=True
)

matcher = ImageMatcher(config)
result = matcher.compare("img1.jpg", "img2.jpg")
```

---

## Performance Tips

1. **Use GPU**: If you have a CUDA GPU, use `device=Device.GPU` for faster processing
2. **Reduce max_image_size**: Smaller images process faster, at slight accuracy cost
3. **Reduce max_features**: Fewer features = faster but may miss some matches
4. **Batch processing**: When comparing many images, reuse the matcher instance

---

## Limitations

- Very heavy filters (like artistic style transfer) may cause false negatives
- Extreme crops (< 10% of original) may not be detected
- Watermark detection works best for text/logos in corners

---

## License

MIT License - feel free to use in your projects.

---

## Credits

Written at **[Sylent.co](https://sylent.co)**

Built with OpenCV, NumPy, and optional CuPy for GPU acceleration.
