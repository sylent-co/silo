#!/usr/bin/env python3
"""
Example usage of SILO - Spatial Iterative Latent Outset

This script demonstrates how to use SILO for image comparison.
Run it from the command line: python example.py
"""

from silo import compare, is_match, Device, ImageMatcher, Config


def basic_comparison():
    """Simple comparison between two images."""
    print("=" * 60)
    print("BASIC COMPARISON")
    print("=" * 60)
    
    result = compare("bird.jpg", "bird2.jpg")
    
    print(f"Match: {result.is_match}")
    print(f"Similarity: {result.similarity:.1%}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Time: {result.processing_time:.3f}s")
    print()


def detailed_comparison():
    """Comparison with detailed output."""
    print("=" * 60)
    print("DETAILED COMPARISON")
    print("=" * 60)
    
    result = compare(
        "sdy/img11.jpg", 
        "sdy/img22.jpg",
        verbose=True  # Print progress
    )
    
    print(f"\nResult: {'MATCH' if result.is_match else 'NO MATCH'}")
    print(f"Method: {result.method}")
    print(f"SSIM: {result.ssim_score:.4f}")
    print(f"Template: {result.template_score:.4f}")
    print(f"Features: {result.feature_count}")
    
    # Check transforms
    t = result.transforms
    if t.is_rotated:
        print(f"Rotation: {t.rotation_angle}°")
    if t.is_cropped:
        print("Cropped: Yes")
    if t.has_filter:
        print(f"Filter: {t.filter_type if t.filter_type else 'Yes'}")
    print()


def quick_check():
    """Quick boolean check if images match."""
    print("=" * 60)
    print("QUICK CHECK")
    print("=" * 60)
    
    pairs = [
        ("bird.jpg", "bird2.jpg"),
        ("sdy/img11.jpg", "sdy/img33.jpg"),
        ("sdy/img11.jpg", "sdy/gg.png"),
    ]
    
    for img1, img2 in pairs:
        match = is_match(img1, img2)
        print(f"{img1} vs {img2}: {'MATCH' if match else 'DIFFERENT'}")
    print()


def gpu_acceleration():
    """Use GPU for faster processing."""
    print("=" * 60)
    print("GPU ACCELERATION")
    print("=" * 60)
    
    # Check if GPU is available
    result = compare("bird.jpg", "bird2.jpg", device=Device.GPU)
    
    print(f"Device used: {result.device}")
    print(f"Time: {result.processing_time:.3f}s")
    print()


def custom_config():
    """Use custom configuration."""
    print("=" * 60)
    print("CUSTOM CONFIGURATION")
    print("=" * 60)
    
    config = Config(
        device=Device.AUTO,
        max_image_size=800,      # Smaller = faster
        max_features=300,        # Fewer features = faster
        verbose=True
    )
    
    matcher = ImageMatcher(config)
    result = matcher.compare("bird.jpg", "bird2.jpg")
    
    print(f"Match: {result.is_match}")
    print()


def batch_comparison():
    """Compare multiple images efficiently."""
    print("=" * 60)
    print("BATCH COMPARISON")
    print("=" * 60)
    
    # Reuse matcher for efficiency
    matcher = ImageMatcher()
    
    images = ["bird.jpg", "bird2.jpg", "sdy/img11.jpg", "sdy/img33.jpg"]
    
    print("Comparing all pairs:")
    for i, img1 in enumerate(images):
        for img2 in images[i+1:]:
            result = matcher.compare(img1, img2)
            status = "MATCH" if result.is_match else "DIFFERENT"
            print(f"  {img1} vs {img2}: {status} ({result.similarity:.1%})")
    print()


if __name__ == "__main__":
    print("\nSILO - Spatial Iterative Latent Outset")
    print("Image Comparison Examples\n")
    
    # Run examples
    basic_comparison()
    detailed_comparison()
    quick_check()
    
    # Uncomment to try these:
    # gpu_acceleration()
    # custom_config()
    # batch_comparison()
    
    print("Done! Check the README.md for more usage examples.")
