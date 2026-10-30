"""Tests for `graspkit.utils.image_utils`."""

from graspkit.utils.image_utils import screen_to_image_pixel


class TestScreenToImagePixel:
    def test_screen_to_image_pixel(self):
        assert screen_to_image_pixel((0.0, 0.0), (360, 640)) == (0, 0)
        assert screen_to_image_pixel((0.5, 0.5), (360, 640)) == (320, 180)
        assert screen_to_image_pixel((1.0, 1.0), (360, 640)) == (639, 359)
