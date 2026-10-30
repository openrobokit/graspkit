def screen_to_image_pixel(screen_position: tuple[float, float], image_shape: tuple[int, int]) -> tuple[int, int]:
    """Map Viser's normalized OpenCV screen position to a clamped image ``(x, y)`` pixel."""
    height, width = image_shape
    x = min(max(int(screen_position[0] * width), 0), width - 1)
    y = min(max(int(screen_position[1] * height), 0), height - 1)
    return x, y
