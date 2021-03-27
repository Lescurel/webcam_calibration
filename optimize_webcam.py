import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread, imshow
from skimage import measure
from scipy.optimize import dual_annealing
import numpy as np
import cv2
import subprocess


def get_brgbw_points(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_g = rgb2gray(img)
    contours = measure.find_contours(img_g, 0.8)
    if not contours:
        raise ValueError("no contours found in the image")
    # ascending sort, biggest contour is the last one  
    roi_contour = sorted(contours, key=lambda x: x.shape[0])[-1]
    # we assume that the phone is in vertical position
    top, left = np.min(roi_contour[:, 0]), np.min(roi_contour[:, 1])
    bottom, right = np.max(roi_contour[:, 0]), np.max(roi_contour[:, 1])
    center = int(left + (right - left) / 2)
    height = bottom - top

    black = (center, int(top + 0.1 * height))
    red = (center, int(top + 0.3 * height))
    green = (center, int(top + 0.5 * height))
    blue = (center, int(top + 0.7 * height))
    white = (center, int(top + 0.9 * height))

    return black, red, green, blue, white


class WebCam:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def get_next_image(self):
        ret, img = self.cam.read()
        if ret:
            return img
        raise EnvironmentError("webcam error")


def set_webcam_parameters(
    brightness, contrast, saturation, hue, gamma, white_balance_temperature
):
    """
    [nix-shell:~/workspace/webcam_calibration]$ v4l2-ctl -l
                         brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                           contrast 0x00980901 (int)    : min=0 max=95 step=1 default=0 value=0
                         saturation 0x00980902 (int)    : min=0 max=100 step=1 default=64 value=64
                                hue 0x00980903 (int)    : min=-2000 max=2000 step=1 default=0 value=0
     white_balance_temperature_auto 0x0098090c (bool)   : default=1 value=0
                              gamma 0x00980910 (int)    : min=100 max=300 step=1 default=100 value=100
               power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=1
          white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4600 value=4600
                          sharpness 0x0098091b (int)    : min=1 max=7 step=1 default=2 value=2
             backlight_compensation 0x0098091c (int)    : min=0 max=3 step=1 default=3 value=3
                      exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
                  exposure_absolute 0x009a0902 (int)    : min=10 max=626 step=1 default=156 value=156 flags=inactive
             exposure_auto_priority 0x009a0903 (bool)   : default=0 value=1

    """
    # avoiding forbidden values
    brightness = int(min(max(-64, brightness), 64))
    contrast = int(min(max(0, contrast), 95))
    saturation = int(min(max(0, saturation), 100))
    hue = int(min(max(-2000, hue), 2000))
    gamma = int(min(max(100, gamma), 300))
    white_balance_temperature = int(min(max(2800, white_balance_temperature), 6500))
    subprocess.run(
        [
            "v4l2-ctl",
            "-c",
            f"brightness={brightness}",
            "-c",
            f"contrast={contrast}",
            "-c",
            f"saturation={saturation}",
            "-c",
            f"hue={hue}",
            "-c",
            f"gamma={gamma}",
            "-c",
            f"white_balance_temperature={white_balance_temperature}",
        ]
    )


def ground_truth():
    return (
        np.array(
            [
                0,
                0,
                0,  # black
                0,
                0,
                255,  # red (opencv works in BGR)
                0,
                255,
                0,  # green
                255,
                0,
                0,  # blue
                255,
                255,
                255,  # white
            ]
        )
        / 255.0
    )


def get_colors(image, points):
    return np.reshape(np.array([image[p[1], p[0]] for p in points]), -1) / 255.0


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def func_to_optimize(x, camera:WebCam, points):
    set_webcam_parameters(*x)
    img = camera.get_next_image()
    return mse(get_colors(img, points), ground_truth())

  

if "__main__" == __name__:
    default_x0 = (
        0,
        0,
        64,
        0,
        100,
        4600
    )
    set_webcam_parameters(*default_x0)
    camera = WebCam()
    img = camera.get_next_image()
    points = get_brgbw_points(img)

    ret = dual_annealing(func=func_to_optimize,
                   bounds=[(-64,64),(0,95),(0,100),(-2000,2000),(100,300),(2800,6500)],
                   args=(camera, points),
                   x0 = default_x0,
    )
    print(ret.success, ret.message)
    print(ret.x)
