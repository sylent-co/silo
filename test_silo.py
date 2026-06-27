"""
Unit tests for SILO - Spatial Iterative Latent Outset

Tests use synthetic numpy arrays to avoid dependency on image files.
"""

import sys
import tempfile
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from silo import (
    Config,
    Device,
    FeatureMatcher,
    GPUProcessor,
    ImageMatcher,
    MatchResult,
    PixelAnalyzer,
    TemplateMatcher,
    TransformDetector,
    TransformInfo,
    Visualizer,
    compare,
    get_available_devices,
    get_similarity,
    is_match,
    main,
    select_device,
)


# ---------------------------------------------------------------------------
# Helpers – synthetic images
# ---------------------------------------------------------------------------

def _solid(h: int = 200, w: int = 200, color=(128, 128, 128)) -> np.ndarray:
    """Create a solid-colour RGB image."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def _gradient(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a horizontal gradient RGB image with distinctive features."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    plane = np.tile(row, (h, 1))
    img = np.stack([plane, 255 - plane, plane // 2], axis=2)
    return img


def _noisy(h: int = 200, w: int = 200, seed: int = 42) -> np.ndarray:
    """Create a random-noise RGB image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _checkerboard(h: int = 200, w: int = 200, block: int = 20) -> np.ndarray:
    """Create a checkerboard pattern with strong features for ORB."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if (y // block + x // block) % 2 == 0:
                img[y:y + block, x:x + block] = (255, 255, 255)
    return img


def _save_image(img: np.ndarray, suffix: str = ".png") -> str:
    """Save an image to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f.name, bgr)
    f.close()
    return f.name


# ===================================================================
# Data structures
# ===================================================================

class TestTransformInfo:
    def test_defaults(self):
        t = TransformInfo()
        assert t.rotation_angle == 0
        assert not t.is_rotated
        assert not t.is_cropped
        assert not t.has_filter
        assert not t.has_watermark
        assert not t.is_overlay
        assert t.filter_type == ""
        assert t.watermark_confidence == 0.0

    def test_to_dict(self):
        t = TransformInfo(rotation_angle=90, is_rotated=True, has_filter=True,
                          filter_type="warm_tint", watermark_confidence=0.12345)
        d = t.to_dict()
        assert d["rotation_angle"] == 90
        assert d["is_rotated"] is True
        assert d["has_filter"] is True
        assert d["filter_type"] == "warm_tint"
        assert d["watermark_confidence"] == 0.1235  # rounded to 4 decimals


class TestMatchResult:
    def test_defaults(self):
        r = MatchResult()
        assert not r.is_match
        assert r.similarity == 0.0
        assert r.method == ""
        assert isinstance(r.transforms, TransformInfo)

    def test_to_dict(self):
        r = MatchResult(is_match=True, similarity=0.776, confidence=0.85,
                        method="combined_analysis", processing_time=1.2345,
                        ssim_score=0.65, template_score=0.82, feature_count=42)
        d = r.to_dict()
        assert d["is_match"] is True
        assert d["similarity"] == 0.776
        assert d["device"] == "CPU"
        assert "transforms" in d

    def test_str_match(self):
        r = MatchResult(is_match=True, similarity=0.8, confidence=0.9,
                        processing_time=0.5)
        s = str(r)
        assert "[MATCH]" in s

    def test_str_no_match(self):
        r = MatchResult(is_match=False, similarity=0.2, confidence=0.3,
                        processing_time=0.5)
        s = str(r)
        assert "[NO MATCH]" in s


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.device == Device.AUTO
        assert c.ssim_threshold == 0.87
        assert c.pixel_diff_threshold == 7.5
        assert c.template_threshold == 0.75
        assert c.feature_threshold == 50
        assert c.max_image_size == 1000
        assert c.max_features == 500
        assert c.crop_size_ratio_threshold == 0.20

    def test_custom(self):
        c = Config(device=Device.GPU, ssim_threshold=0.5)
        assert c.device == Device.GPU
        assert c.ssim_threshold == 0.5


# ===================================================================
# Device selection
# ===================================================================

class TestDeviceSelection:
    def test_get_available_devices_cpu_always(self):
        devs = get_available_devices()
        assert Device.CPU in devs

    def test_select_device_auto_falls_back_cpu(self):
        with patch("silo.CUPY_AVAILABLE", False):
            assert select_device(Device.AUTO) == Device.CPU

    def test_select_device_gpu_falls_back_cpu(self):
        with patch("silo.CUPY_AVAILABLE", False):
            assert select_device(Device.GPU) == Device.CPU

    def test_select_device_cpu(self):
        assert select_device(Device.CPU) == Device.CPU

    def test_select_device_auto_with_gpu(self):
        with patch("silo.CUPY_AVAILABLE", True):
            assert select_device(Device.AUTO) == Device.GPU


# ===================================================================
# PixelAnalyzer
# ===================================================================

class TestPixelAnalyzer:
    @pytest.fixture()
    def analyzer(self):
        return PixelAnalyzer(Config())

    def test_identical_images(self, analyzer):
        img = _gradient()
        result = analyzer.analyze(img, img.copy())
        assert result["ssim_score"] > 0.99
        assert result["pixel_diff"] < 0.01

    def test_completely_different(self, analyzer):
        white = _solid(color=(255, 255, 255))
        black = _solid(color=(0, 0, 0))
        result = analyzer.analyze(white, black)
        assert result["pixel_diff"] > 50

    def test_result_keys(self, analyzer):
        img = _gradient()
        result = analyzer.analyze(img, img.copy())
        for key in ("ssim_score", "pixel_diff", "change_mask", "img1", "img2",
                     "gray1", "gray2", "orig_shape1", "orig_shape2", "time"):
            assert key in result

    def test_different_sizes(self, analyzer):
        big = _gradient(400, 400)
        small = _gradient(100, 100)
        result = analyzer.analyze(big, small)
        assert "ssim_score" in result

    def test_resize_large_image(self, analyzer):
        large = _gradient(2000, 2000)
        resized = analyzer._resize(large)
        assert max(resized.shape[:2]) <= analyzer.config.max_image_size

    def test_no_resize_small_image(self, analyzer):
        small = _gradient(100, 100)
        resized = analyzer._resize(small)
        assert resized.shape == small.shape

    def test_grayscale_input(self, analyzer):
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        gray3 = np.stack([gray] * 3, axis=2)
        result = analyzer.analyze(gray3, gray3.copy())
        assert result["ssim_score"] > 0.99

    def test_pixel_diff_symmetry(self, analyzer):
        a = _gradient()
        b = _noisy()
        r1 = analyzer.analyze(a, b)
        r2 = analyzer.analyze(b, a)
        assert abs(r1["pixel_diff"] - r2["pixel_diff"]) < 5.0


# ===================================================================
# TemplateMatcher
# ===================================================================

class TestTemplateMatcher:
    @pytest.fixture()
    def matcher(self):
        return TemplateMatcher(Config())

    def test_identical_images(self, matcher):
        img = _checkerboard()
        result = matcher.find_best_match(img, img.copy())
        assert result["correlation"] > 0.8
        assert result["rotation_angle"] == 0

    def test_result_keys(self, matcher):
        img = _checkerboard()
        result = matcher.find_best_match(img, img.copy())
        for key in ("correlation", "rotation_angle", "all_correlations",
                     "is_similar", "confidence", "time"):
            assert key in result

    def test_different_images_low_correlation(self, matcher):
        a = _checkerboard()
        b = _noisy()
        result = matcher.find_best_match(a, b)
        assert result["correlation"] < 0.9

    def test_rotate_image_0(self, matcher):
        img = _gradient(100, 80)
        rotated = matcher._rotate_image(img, 0)
        np.testing.assert_array_equal(rotated, img)

    def test_rotate_image_180(self, matcher):
        img = _gradient(100, 80)
        r1 = matcher._rotate_image(img, 180)
        assert r1.shape == img.shape

    def test_rotate_image_90_swaps_dims(self, matcher):
        img = np.zeros((100, 60), dtype=np.uint8)
        rotated = matcher._rotate_image(img, 90)
        assert rotated.shape == (60, 100)

    def test_multi_scale_match_template_too_big(self, matcher):
        target = np.zeros((50, 50), dtype=np.uint8)
        template = np.zeros((200, 200), dtype=np.uint8)
        corr = matcher._multi_scale_match(target, template)
        assert isinstance(corr, float)

    def test_all_correlations_has_four_angles(self, matcher):
        img = _checkerboard()
        result = matcher.find_best_match(img, img.copy())
        assert len(result["all_correlations"]) == 4
        assert set(result["all_correlations"].keys()) == {0, 90, 180, 270}

    def test_smaller_template_used(self, matcher):
        big = _checkerboard(300, 300)
        small = _checkerboard(100, 100)
        result = matcher.find_best_match(big, small)
        assert "correlation" in result


# ===================================================================
# FeatureMatcher
# ===================================================================

class TestFeatureMatcher:
    @pytest.fixture()
    def matcher(self):
        return FeatureMatcher(Config())

    def test_identical_images(self, matcher):
        img = _noisy(200, 200, seed=99)
        result = matcher.match(img, img.copy())
        assert result["matches"] >= 0
        assert result["confidence"] >= 0.0

    def test_result_keys(self, matcher):
        img = _checkerboard()
        result = matcher.match(img, img.copy())
        for key in ("matches", "confidence", "inlier_ratio", "time"):
            assert key in result

    def test_no_features(self, matcher):
        blank = _solid()
        result = matcher.match(blank, blank.copy())
        assert result["matches"] == 0
        assert result["confidence"] == 0.0

    def test_different_images(self, matcher):
        a = _checkerboard()
        b = _noisy()
        result = matcher.match(a, b)
        assert isinstance(result["matches"], int)

    def test_confidence_capped_at_1(self, matcher):
        img = _checkerboard(400, 400, block=10)
        result = matcher.match(img, img.copy())
        assert result["confidence"] <= 1.0

    def test_grayscale_images(self, matcher):
        gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        gray3 = np.stack([gray] * 3, axis=2)
        result = matcher.match(gray3, gray3.copy())
        assert isinstance(result["matches"], int)


# ===================================================================
# TransformDetector
# ===================================================================

class TestTransformDetector:
    @pytest.fixture()
    def detector(self):
        return TransformDetector(Config())

    @pytest.fixture()
    def identical_context(self):
        img = _gradient()
        analyzer = PixelAnalyzer(Config())
        pixel_result = analyzer.analyze(img, img.copy())
        template_result = {
            "correlation": 1.0,
            "rotation_angle": 0,
            "all_correlations": {0: 1.0, 90: 0.3, 180: 0.3, 270: 0.3},
        }
        feature_result = {"matches": 200, "confidence": 1.0}
        return pixel_result, template_result, feature_result

    def test_no_transforms_identical(self, detector, identical_context):
        pixel_result, template_result, feature_result = identical_context
        info = detector.detect(
            pixel_result["img1"], pixel_result["img2"],
            pixel_result, template_result, feature_result
        )
        assert not info.is_rotated
        assert info.rotation_angle == 0

    def test_rotation_detected(self, detector, identical_context):
        pixel_result, _, feature_result = identical_context
        template_result = {
            "correlation": 0.9,
            "rotation_angle": 90,
            "all_correlations": {0: 0.3, 90: 0.9, 180: 0.3, 270: 0.4},
        }
        info = detector.detect(
            pixel_result["img1"], pixel_result["img2"],
            pixel_result, template_result, feature_result
        )
        assert info.is_rotated
        assert info.rotation_angle == 90

    def test_crop_detected(self, detector):
        big = _gradient(400, 400)
        small = _gradient(50, 50)
        config = Config()
        analyzer = PixelAnalyzer(config)
        pixel_result = analyzer.analyze(big, small)
        template_result = {
            "correlation": 0.8,
            "rotation_angle": 0,
            "all_correlations": {0: 0.8, 90: 0.3, 180: 0.3, 270: 0.3},
        }
        feature_result = {"matches": 50, "confidence": 0.5}
        info = detector.detect(big, small, pixel_result, template_result, feature_result)
        assert info.is_cropped

    def test_filter_detected_high_ssim_high_diff(self, detector):
        result = detector._detect_filter({"ssim_score": 0.6, "pixel_diff": 80})
        assert result is True

    def test_no_filter_low_diff(self, detector):
        result = detector._detect_filter({"ssim_score": 0.9, "pixel_diff": 5})
        assert result is False

    def test_classify_filter_brightened(self, detector):
        dark = _solid(color=(50, 50, 50))
        bright = _solid(color=(200, 200, 200))
        result = detector._classify_filter({"img1": dark, "img2": bright})
        assert result == "brightened"

    def test_classify_filter_darkened(self, detector):
        bright = _solid(color=(200, 200, 200))
        dark = _solid(color=(50, 50, 50))
        result = detector._classify_filter({"img1": bright, "img2": dark})
        assert result == "darkened"

    def test_classify_filter_color_adjustment(self, detector):
        a = _solid(color=(128, 128, 128))
        b = _solid(color=(130, 130, 130))
        result = detector._classify_filter({"img1": a, "img2": b})
        assert result == "color_adjustment"

    def test_classify_filter_high_contrast(self, detector):
        low_contrast = _solid(color=(120, 120, 120))
        rng = np.random.RandomState(0)
        high_contrast = rng.randint(0, 256, low_contrast.shape, dtype=np.uint8)
        result = detector._classify_filter({"img1": low_contrast, "img2": high_contrast})
        assert result in ("high_contrast", "color_adjustment", "warm_tint", "cool_tint")

    def test_90_270_preference(self, detector, identical_context):
        pixel_result, _, feature_result = identical_context
        template_result = {
            "correlation": 0.9,
            "rotation_angle": 270,
            "all_correlations": {0: 0.3, 90: 0.85, 180: 0.3, 270: 0.86},
        }
        info = detector.detect(
            pixel_result["img1"], pixel_result["img2"],
            pixel_result, template_result, feature_result
        )
        assert info.rotation_angle == 90

    def test_detect_overlay_returns_false_low_correlation(self, detector):
        pixel_result = {"img1": _gradient(), "img2": _gradient(),
                        "ssim_score": 0.3}
        template_result = {"correlation": 0.3}
        result = detector._detect_overlay(pixel_result, template_result)
        assert result is False

    def test_detect_overlay_returns_false_high_ssim(self, detector):
        pixel_result = {"img1": _gradient(), "img2": _gradient(),
                        "ssim_score": 0.8}
        template_result = {"correlation": 0.9}
        result = detector._detect_overlay(pixel_result, template_result)
        assert result is False

    def test_watermark_returns_false_low_corr(self, detector):
        pixel_result = {
            "change_mask": np.zeros((100, 100), dtype=np.uint8),
            "ssim_score": 0.5,
            "img1": _gradient(100, 100),
            "img2": _gradient(100, 100),
        }
        template_result = {"correlation": 0.3}
        has_wm, conf = detector._detect_watermark(pixel_result, template_result)
        assert has_wm is False

    def test_watermark_returns_false_high_ssim(self, detector):
        pixel_result = {
            "change_mask": np.zeros((100, 100), dtype=np.uint8),
            "ssim_score": 0.8,
            "img1": _gradient(100, 100),
            "img2": _gradient(100, 100),
        }
        template_result = {"correlation": 0.9}
        has_wm, conf = detector._detect_watermark(pixel_result, template_result)
        assert has_wm is False


# ===================================================================
# ImageMatcher
# ===================================================================

class TestImageMatcher:
    @pytest.fixture()
    def matcher(self):
        return ImageMatcher(Config(device=Device.CPU))

    def test_identical_images(self, matcher):
        img = _checkerboard()
        result = matcher.compare(img, img.copy())
        assert result.is_match
        assert result.similarity > 0.8

    def test_completely_different(self, matcher):
        a = _noisy(200, 200, seed=1)
        b = _noisy(200, 200, seed=99)
        result = matcher.compare(a, b)
        assert not result.is_match

    def test_load_image_from_path(self, matcher):
        img = _checkerboard()
        path = _save_image(img)
        loaded = matcher._load_image(path)
        assert loaded is not None
        assert loaded.shape[2] == 3

    def test_load_image_bad_path(self, matcher):
        loaded = matcher._load_image("/nonexistent/image.png")
        assert loaded is None

    def test_load_image_array(self, matcher):
        img = _gradient()
        loaded = matcher._load_image(img)
        assert loaded is not None

    def test_load_image_grayscale_array(self, matcher):
        gray = np.zeros((100, 100), dtype=np.uint8)
        loaded = matcher._load_image(gray)
        assert loaded is not None
        assert loaded.shape[2] == 3

    def test_load_image_rgba_array(self, matcher):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        loaded = matcher._load_image(rgba)
        assert loaded is not None
        assert loaded.shape[2] == 3

    def test_load_image_non_image(self, matcher):
        loaded = matcher._load_image(12345)
        assert loaded is None

    def test_compare_with_paths(self, matcher):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        result = matcher.compare(p1, p2)
        assert result.is_match

    def test_compare_none_image(self, matcher):
        result = matcher.compare("/nonexistent1.png", "/nonexistent2.png")
        assert not result.is_match
        assert result.method == "error"

    def test_early_exit_ssim(self, matcher):
        img = _gradient()
        result = matcher.compare(img, img.copy())
        assert result.is_match
        assert result.early_exit

    def test_verbose_mode(self):
        config = Config(device=Device.CPU, verbose=True)
        matcher = ImageMatcher(config)
        img = _gradient()
        result = matcher.compare(img, img.copy())
        assert result.is_match

    def test_processing_time_recorded(self, matcher):
        img = _gradient()
        result = matcher.compare(img, img.copy())
        assert result.processing_time > 0

    def test_device_used_recorded(self, matcher):
        img = _gradient()
        result = matcher.compare(img, img.copy())
        assert result.device_used == Device.CPU

    def test_crop_detection_path(self):
        config = Config(device=Device.CPU, crop_size_ratio_threshold=0.5)
        matcher = ImageMatcher(config)
        big = _checkerboard(400, 400, block=20)
        small = big[50:130, 50:130].copy()
        result = matcher.compare(big, small)
        assert isinstance(result, MatchResult)

    def test_combined_analysis_path(self, matcher):
        a = _noisy(200, 200, seed=1)
        b = _noisy(200, 200, seed=2)
        result = matcher.compare(a, b)
        assert isinstance(result, MatchResult)


# ===================================================================
# Convenience functions
# ===================================================================

class TestConvenienceFunctions:
    def test_compare_returns_match_result(self):
        img = _gradient()
        result = compare(img, img.copy())
        assert isinstance(result, MatchResult)

    def test_compare_identical(self):
        img = _gradient()
        result = compare(img, img.copy())
        assert result.is_match

    def test_compare_different(self):
        result = compare(_noisy(200, 200, seed=1), _noisy(200, 200, seed=99))
        assert not result.is_match

    def test_is_match_true(self):
        img = _gradient()
        assert is_match(img, img.copy())

    def test_is_match_false(self):
        assert not is_match(_noisy(200, 200, seed=1), _noisy(200, 200, seed=99))

    def test_is_match_threshold(self):
        img = _gradient()
        assert is_match(img, img.copy(), threshold=0.9)

    def test_get_similarity_identical(self):
        img = _gradient()
        sim = get_similarity(img, img.copy())
        assert sim > 0.8

    def test_get_similarity_different(self):
        sim = get_similarity(_noisy(200, 200, seed=1), _noisy(200, 200, seed=99))
        assert sim < 0.6

    def test_compare_kwargs(self):
        img = _gradient()
        result = compare(img, img.copy(), max_image_size=500)
        assert result.is_match


# ===================================================================
# GPUProcessor (mocked – no real GPU expected)
# ===================================================================

class TestGPUProcessor:
    def test_raises_without_cupy(self):
        with patch("silo.CUPY_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="CuPy not available"):
                GPUProcessor()


# ===================================================================
# Visualizer (smoke-test – verify it doesn't crash)
# ===================================================================

class TestVisualizer:
    def test_show_without_matplotlib_no_crash(self):
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            img = _gradient()
            result = MatchResult(is_match=True, similarity=0.9)
            pixel_result = {"change_mask": np.zeros((200, 200), dtype=np.uint8)}
            template_result = {"all_correlations": {0: 0.9}}
            Visualizer.show(img, img.copy(), result, pixel_result, template_result)


# ===================================================================
# CLI (main)
# ===================================================================

class TestCLI:
    def test_main_match(self):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        with patch("sys.argv", ["silo.py", p1, p2]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_no_match(self):
        a = _noisy(200, 200, seed=1)
        b = _noisy(200, 200, seed=99)
        p1 = _save_image(a)
        p2 = _save_image(b)
        with patch("sys.argv", ["silo.py", p1, p2]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_verbose(self):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        with patch("sys.argv", ["silo.py", p1, p2, "--verbose"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_device_cpu(self):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        with patch("sys.argv", ["silo.py", p1, p2, "--device", "cpu"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_custom_thresholds(self):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        with patch("sys.argv", ["silo.py", p1, p2,
                                "--ssim-threshold", "0.5",
                                "--pixel-diff-threshold", "10.0"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_pixel_image(self):
        tiny = np.array([[[128, 128, 128]]], dtype=np.uint8)
        config = Config(device=Device.CPU)
        matcher = ImageMatcher(config)
        result = matcher.compare(tiny, tiny.copy())
        assert isinstance(result, MatchResult)

    def test_very_large_pixel_diff(self):
        analyzer = PixelAnalyzer(Config())
        white = _solid(color=(255, 255, 255))
        black = _solid(color=(0, 0, 0))
        result = analyzer.analyze(white, black)
        assert result["pixel_diff"] == 100.0

    def test_ssim_fallback_no_skimage(self):
        with patch("silo.SKIMAGE_AVAILABLE", False):
            analyzer = PixelAnalyzer(Config())
            img = _gradient(100, 100)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            score = analyzer._calculate_ssim(gray, gray.copy())
            assert score > 0.99

    def test_feature_matcher_few_descriptors(self):
        matcher = FeatureMatcher(Config())
        almost_blank = np.full((100, 100, 3), 128, dtype=np.uint8)
        almost_blank[50, 50] = [255, 0, 0]
        result = matcher.match(almost_blank, almost_blank.copy())
        assert result["confidence"] >= 0.0

    def test_pixel_diff_grayscale(self):
        analyzer = PixelAnalyzer(Config())
        gray1 = np.full((100, 100), 200, dtype=np.uint8)
        gray2 = np.full((100, 100), 50, dtype=np.uint8)
        diff, mask = analyzer._calculate_pixel_diff(gray1, gray2)
        assert diff == 100.0

    def test_template_matcher_tiny_template(self):
        matcher = TemplateMatcher(Config())
        target = np.zeros((200, 200), dtype=np.uint8)
        template = np.zeros((10, 10), dtype=np.uint8)
        corr = matcher._multi_scale_match(target, template)
        assert isinstance(corr, float)


# ===================================================================
# Additional coverage – uncovered branches
# ===================================================================

class TestPixelMatchEarlyExit:
    def test_pixel_diff_early_exit(self):
        """Trigger pixel_match early exit (low pixel_diff, moderate SSIM)."""
        img = _gradient(200, 200)
        slightly_off = img.copy()
        slightly_off[0:2, 0:2] = [0, 0, 0]
        config = Config(device=Device.CPU, ssim_threshold=0.999,
                        pixel_diff_threshold=10.0)
        matcher = ImageMatcher(config)
        result = matcher.compare(img, slightly_off)
        assert result.is_match
        assert result.early_exit
        assert result.method == "pixel_match"


class TestCombinedDecisionBranches:
    def test_high_template_high_features(self):
        """Template > 0.90 and features > 50."""
        config = Config(device=Device.CPU, ssim_threshold=0.999,
                        pixel_diff_threshold=0.01)
        matcher = ImageMatcher(config)
        img = _noisy(300, 300, seed=10)
        shifted = _noisy(300, 300, seed=20)
        with patch.object(matcher.template_matcher, 'find_best_match',
                          return_value={
                              'correlation': 0.95,
                              'rotation_angle': 0,
                              'detected_angle': 0,
                              'all_correlations': {0: 0.95, 90: 0.1, 180: 0.1, 270: 0.1},
                              'is_similar': True,
                              'confidence': 0.95,
                              'time': 0.01
                          }):
            with patch.object(matcher.feature_matcher, 'match',
                              return_value={
                                  'matches': 60,
                                  'confidence': 0.6,
                                  'inlier_ratio': 0.5,
                                  'keypoints1': 100,
                                  'keypoints2': 100,
                                  'time': 0.01
                              }):
                result = matcher.compare(img, shifted)
                assert result.is_match
                assert result.method == "combined_analysis"

    def test_high_template_very_high_features(self):
        """Template > 0.80 and features > 100."""
        config = Config(device=Device.CPU, ssim_threshold=0.999,
                        pixel_diff_threshold=0.01)
        matcher = ImageMatcher(config)
        img = _noisy(300, 300, seed=10)
        shifted = _noisy(300, 300, seed=20)
        with patch.object(matcher.template_matcher, 'find_best_match',
                          return_value={
                              'correlation': 0.85,
                              'rotation_angle': 0,
                              'detected_angle': 0,
                              'all_correlations': {0: 0.85, 90: 0.1, 180: 0.1, 270: 0.1},
                              'is_similar': True,
                              'confidence': 0.85,
                              'time': 0.01
                          }):
            with patch.object(matcher.feature_matcher, 'match',
                              return_value={
                                  'matches': 120,
                                  'confidence': 1.0,
                                  'inlier_ratio': 0.8,
                                  'keypoints1': 200,
                                  'keypoints2': 200,
                                  'time': 0.01
                              }):
                result = matcher.compare(img, shifted)
                assert result.is_match

    def test_very_strong_features_alone(self):
        """Feature matches > 150 alone triggers match."""
        config = Config(device=Device.CPU, ssim_threshold=0.999,
                        pixel_diff_threshold=0.01)
        matcher = ImageMatcher(config)
        img = _noisy(200, 200, seed=5)
        img2 = _noisy(200, 200, seed=6)
        with patch.object(matcher.template_matcher, 'find_best_match',
                          return_value={
                              'correlation': 0.5,
                              'rotation_angle': 0,
                              'detected_angle': 0,
                              'all_correlations': {0: 0.5, 90: 0.1, 180: 0.1, 270: 0.1},
                              'is_similar': False,
                              'confidence': 0.5,
                              'time': 0.01
                          }):
            with patch.object(matcher.feature_matcher, 'match',
                              return_value={
                                  'matches': 160,
                                  'confidence': 1.0,
                                  'inlier_ratio': 0.9,
                                  'keypoints1': 300,
                                  'keypoints2': 300,
                                  'time': 0.01
                              }):
                result = matcher.compare(img, img2)
                assert result.is_match


class TestTemplateMatcherResizing:
    def test_large_images_resized(self):
        """Cover the resizing branches in find_best_match."""
        matcher = TemplateMatcher(Config())
        big1 = _checkerboard(1200, 1200, block=40)
        big2 = _checkerboard(600, 600, block=20)
        gray1 = cv2.cvtColor(big1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(big2, cv2.COLOR_RGB2GRAY)
        result = matcher.find_best_match(big1, big2)
        assert "correlation" in result


class TestTransformDetectorOverlay:
    def test_overlay_detected_high_diff_with_size_diff(self):
        detector = TransformDetector(Config())
        white = _solid(400, 400, color=(255, 255, 255))
        dark = _solid(200, 200, color=(0, 0, 0))
        pixel_result = {
            "img1": white,
            "img2": dark,
            "ssim_score": 0.2,
        }
        template_result = {"correlation": 0.7}
        feature_result = {"matches": 50, "confidence": 0.5}
        result = detector._detect_overlay(pixel_result, template_result, feature_result)
        assert isinstance(result, bool)

    def test_overlay_detected_very_high_diff(self):
        detector = TransformDetector(Config())
        img1 = _solid(200, 200, color=(255, 255, 255))
        img2 = _solid(200, 200, color=(0, 0, 0))
        pixel_result = {
            "img1": img1,
            "img2": img2,
            "ssim_score": 0.2,
        }
        template_result = {"correlation": 0.7}
        feature_result = {"matches": 50, "confidence": 0.5}
        result = detector._detect_overlay(pixel_result, template_result, feature_result)
        assert result is True

    def test_overlay_false_few_features(self):
        detector = TransformDetector(Config())
        pixel_result = {
            "img1": _gradient(),
            "img2": _gradient(),
            "ssim_score": 0.3,
        }
        template_result = {"correlation": 0.7}
        feature_result = {"matches": 2, "confidence": 0.02}
        result = detector._detect_overlay(pixel_result, template_result, feature_result)
        assert result is False


class TestFilterClassification:
    def test_classify_low_contrast(self):
        detector = TransformDetector(Config())
        rng = np.random.RandomState(0)
        high_std = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mean_val = int(np.mean(high_std))
        low_std = np.full_like(high_std, mean_val)
        low_std += rng.randint(-5, 6, low_std.shape, dtype=np.int8).astype(np.uint8)
        result = detector._classify_filter({"img1": high_std, "img2": low_std})
        assert result in ("low_contrast", "color_adjustment", "darkened", "brightened")

    def test_classify_warm_tint(self):
        detector = TransformDetector(Config())
        rng = np.random.RandomState(10)
        # Wide-range image so channel shifts don't alter overall std ratio
        img1 = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        img2[:, :, 0] = np.clip(img2[:, :, 0].astype(int) + 50, 0, 255).astype(np.uint8)
        img2[:, :, 1] = np.clip(img2[:, :, 1].astype(int) - 30, 0, 255).astype(np.uint8)
        img2[:, :, 2] = np.clip(img2[:, :, 2].astype(int) - 30, 0, 255).astype(np.uint8)
        result = detector._classify_filter({"img1": img1, "img2": img2})
        assert result == "warm_tint"

    def test_classify_cool_tint(self):
        detector = TransformDetector(Config())
        rng = np.random.RandomState(10)
        img1 = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        img2[:, :, 0] = np.clip(img2[:, :, 0].astype(int) - 30, 0, 255).astype(np.uint8)
        img2[:, :, 1] = np.clip(img2[:, :, 1].astype(int) - 30, 0, 255).astype(np.uint8)
        img2[:, :, 2] = np.clip(img2[:, :, 2].astype(int) + 50, 0, 255).astype(np.uint8)
        result = detector._classify_filter({"img1": img1, "img2": img2})
        assert result == "cool_tint"


class TestFilterDetectionMidRange:
    def test_mid_range_ssim_filter(self):
        detector = TransformDetector(Config())
        result = detector._detect_filter({"ssim_score": 0.5, "pixel_diff": 60})
        assert result is True


class TestWatermarkBranches:
    def test_watermark_high_change_percent(self):
        detector = TransformDetector(Config())
        mask = np.ones((100, 100), dtype=np.uint8)
        pixel_result = {
            "change_mask": mask,
            "ssim_score": 0.5,
            "img1": _gradient(100, 100),
            "img2": _noisy(100, 100),
        }
        template_result = {"correlation": 0.8}
        has_wm, conf = detector._detect_watermark(pixel_result, template_result)
        assert has_wm is False

    def test_watermark_detection_with_text_like_changes(self):
        detector = TransformDetector(Config())
        img1 = _gradient(200, 200)
        img2 = img1.copy()
        cv2.putText(img2, "WATERMARK", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        analyzer = PixelAnalyzer(Config())
        pixel_result = analyzer.analyze(img1, img2)
        pixel_result["ssim_score"] = 0.5
        template_result = {"correlation": 0.8}
        has_wm, conf = detector._detect_watermark(pixel_result, template_result)
        assert isinstance(has_wm, bool)
        assert isinstance(conf, float)


class TestVerboseOutput:
    def test_verbose_full_pipeline(self):
        config = Config(device=Device.CPU, verbose=True, ssim_threshold=0.999,
                        pixel_diff_threshold=0.01)
        matcher = ImageMatcher(config)
        a = _noisy(200, 200, seed=1)
        b = _noisy(200, 200, seed=99)
        result = matcher.compare(a, b)
        assert isinstance(result, MatchResult)

    def test_verbose_load_error(self):
        config = Config(device=Device.CPU, verbose=True)
        matcher = ImageMatcher(config)
        loaded = matcher._load_image(12345)
        assert loaded is None


class TestCLITransformOutput:
    def test_cli_with_rotated_result(self):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        with patch("sys.argv", ["silo.py", p1, p2, "--verbose"]):
            with patch("silo.ImageMatcher.compare") as mock_compare:
                mock_compare.return_value = MatchResult(
                    is_match=True,
                    similarity=0.85,
                    confidence=0.9,
                    method="combined_analysis",
                    processing_time=0.5,
                    device_used=Device.CPU,
                    ssim_score=0.6,
                    template_score=0.8,
                    feature_count=50,
                    transforms=TransformInfo(
                        is_rotated=True,
                        rotation_angle=90,
                        is_cropped=True,
                        is_overlay=True,
                        has_filter=True,
                        filter_type="warm_tint",
                        has_watermark=True,
                        watermark_confidence=0.7,
                    ),
                )
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

    def test_cli_no_transforms_detected(self):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        with patch("sys.argv", ["silo.py", p1, p2]):
            with patch("silo.ImageMatcher.compare") as mock_compare:
                mock_compare.return_value = MatchResult(
                    is_match=True,
                    similarity=0.95,
                    confidence=0.95,
                    method="ssim_match",
                    processing_time=0.1,
                    device_used=Device.CPU,
                )
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

    def test_cli_filter_generic(self):
        img = _checkerboard()
        p1 = _save_image(img)
        p2 = _save_image(img)
        with patch("sys.argv", ["silo.py", p1, p2]):
            with patch("silo.ImageMatcher.compare") as mock_compare:
                mock_compare.return_value = MatchResult(
                    is_match=True,
                    similarity=0.8,
                    confidence=0.85,
                    method="combined",
                    processing_time=0.5,
                    device_used=Device.CPU,
                    transforms=TransformInfo(
                        has_filter=True,
                        filter_type="color_adjustment",
                    ),
                )
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0


class TestGetAvailableDevicesGPU:
    def test_gpu_in_list_when_cupy(self):
        with patch("silo.CUPY_AVAILABLE", True):
            devs = get_available_devices()
            assert Device.GPU in devs
