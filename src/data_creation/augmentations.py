'''
Augmentations to be done
1. Background random crop and resize
2. Background random noise and blur
3. Background random color change
4. Background random rotate and flip
5. Foreground mask smoothing (imp)
6. Foreground mask/image random rotate and flip
7. Foreground mask/image color jitter
8. Foreground mask/image random resize
'''

import cv2
import random
import numpy as np
import imutils
from PIL import ImageOps, Image
import os
from augmentation.warp import Curve, Distort, Stretch
from augmentation.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY, Flip
from augmentation.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from augmentation.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from augmentation.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from augmentation.camera import Contrast, Brightness, JpegCompression, Pixelate
from augmentation.weather import Fog, Snow, Frost, Rain, Shadow
from augmentation.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color


class Augmentations:
    def __init__(self, img):
        self.img = img
        self.h, self.w, _ = self.img.shape

    def crop(self, args_dict):
        x, y = args_dict['crop_x'], args_dict['crop_y']
        ww, hh = args_dict['crop_w'], args_dict['crop_h']
        self.img = self.img[y: y+hh, x: x+ww]

    def resize(self, args_dict):
        w, h = args_dict['resize_w'], args_dict['resize_h']
        self.img = cv2.resize(self.img, (w, h), cv2.INTER_AREA)

    def equalize_histogram(self, args_dict):
        R, G, B = cv2.split(self.img)
        clahe = cv2.createCLAHE(clipLimit=args_dict['eq_hist_clip_limit'], tileGridSize=args_dict['eq_hist_grid_size'])
        output1_R = clahe.apply(R)
        output1_G = clahe.apply(G)
        output1_B = clahe.apply(B)
        self.img = cv2.merge((output1_R, output1_G, output1_B))

    def solarize(self, args_dict):
        img = Image.fromarray(self.img)
        self.img = np.array(ImageOps.solarize(img)).astype(np.uint8)

    def auto_contrast(self, args_dict):
        img = Image.fromarray(self.img)
        red, green, blue = img.split()
        red = np.array(ImageOps.autocontrast(red)).astype(np.uint8)
        green = np.array(ImageOps.autocontrast(green)).astype(np.uint8)
        blue = np.array(ImageOps.autocontrast(blue)).astype(np.uint8)
        self.img = cv2.merge((red, green, blue))

    def noise(self, args_dict):
        noise_typ = args_dict['noise_type']
        if noise_typ == "gauss":
            row, col, ch = self.img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = self.img + gauss
            self.img = noisy.astype(np.uint8)

    def blur(self, args_dict):
        self.img = cv2.blur(self.img, args_dict['blur'])

    def color_jitter(self, args_dict):
        jitter_type = args_dict['jitter_type']
        ch = random.choice([0, 1, 2])
        if jitter_type == '0_channel':
            zeros = np.zeros(self.img.shape[:2], dtype=np.uint8)
            self.img[:, :, ch] = zeros
        elif jitter_type == 'noise_channel':
            noise = np.random.uniform(-100, 100, self.img.shape[:2]).astype(np.uint8)
            ch_img = self.img[:, :, ch]
            ch_img += noise
            self.img[:, :, ch] = ch_img
        elif jitter_type == 'colour_change':
            noise = np.random.randint(-50, 50)
            self.img = self.img.astype(np.int16)
            self.img[:, :, ch] += noise
            self.img = self.img.astype(np.uint8)
        else:
            noise = np.random.uniform(0, 255, self.img.shape[:2]).astype(np.uint8)
            self.img[:, :, ch] = noise

    def rotate(self, args_dict):
        angle = args_dict['angle']
        # angle_range = [-10,  10]
        # angle = np.random.uniform(angle_range[0], angle_range[1])
        self.img = imutils.rotate(self.img, angle)

    def flip(self, args_dict):
        flip_code = args_dict['flip_code']
        self.img = cv2.flip(self.img, flip_code)

    def erosion(self, args_dict):
        kernel = args_dict['kernel']
        # kernel = np.ones((random.randint(2, 5), random.randint(2, 5)), np.uint8)
        self.img = cv2.erode(self.img, kernel, iterations=1)

    def dilation(self, args_dict):
        kernel = args_dict['kernel']
        # kernel = np.ones((random.randint(2, 5), random.randint(2, 5)), np.uint8)
        self.img = cv2.dilate(self.img, kernel, iterations=1)

    def get_image(self):
        return self.img


aug_ops = [Curve(), Flip(), Rotate(), Perspective(), Distort(), Stretch(), Shrink(), TranslateX(), TranslateY(), VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]
# aug_ops.extend([GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()])
aug_ops.extend([GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()])
aug_ops.extend([Contrast(), Brightness(), JpegCompression(), Pixelate()])
aug_ops.extend([Fog(), Snow(), Frost(), Rain(), Shadow()])
aug_ops.extend([Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()])


class AugmentImage:
    def __init__(self, img, mask):
        self.img = img
        self.mask = mask
        self.aug_img = Augmentations(self.img)
        if self.mask is not None:
            self.aug_mask = Augmentations(self.mask)
        else:
            self.aug_mask = None

    def get_augments(self):
        aug = self.aug_img
        aug_m = self.aug_mask
        if aug_m is None:
            augments = [
                aug.equalize_histogram,
                aug.solarize,
                aug.auto_contrast,
                aug.erosion,
                aug.dilation,
                aug.noise,
                aug.blur,
                aug.flip,
                aug.color_jitter,
                aug.rotate,
                aug.crop,
                aug.resize,
            ]
        else:
            augments = [
                (aug.resize, aug_m.resize),
                aug.auto_contrast,
                # aug.equalize_histogram,
                # aug.solarize,
                aug.blur,
                (aug.rotate, aug_m.rotate),
                (aug.flip, aug_m.flip),
                aug_m.erosion,
                aug_m.dilation,
                aug.noise,
            ]
        return augments

    def augment_image(self):
        augments = self.get_augments()
        augments = [augment for augment in augments if random.randint(1, 10) < 4]
        # augments = [augments[1]]
        img, mask = None, None
        for augmentions in augments:
            angle = {True: random.randint(-90, 90), False: random.randint(-5, 5)} [self.mask is None]
            flip_code = {True: random.choice([-1, 1, 0]), False: 1} [self.mask is None]
            crop_x = random.randint(10, self.img.shape[1]-10)
            crop_y = random.randint(10, self.img.shape[0]-10)
            crop_w = random.randint(crop_x, self.img.shape[1])
            crop_h = random.randint(crop_y, self.img.shape[0])
            args_dict = {
                'kernel': np.ones((random.randint(2, 3), random.randint(2, 3)), np.uint8),
                'resize_w': random.randint(self.img.shape[1]//2, self.img.shape[1]*2),
                'resize_h': random.randint(self.img.shape[1]//2, self.img.shape[1]*2),
                'flip_code': flip_code,
                'angle': angle,
                'blur': (random.randint(2, 3), random.randint(2, 3)),
                'crop_x': crop_x,
                'crop_y': crop_y,
                'crop_w': crop_w,
                'crop_h': crop_h,
                'noise_type': random.choice(['gauss']),
                'jitter_type': random.choice(['random_channel', 'noise_channel', 'colour_change']),
                'eq_hist_clip_limit': random.randint(1, 10),
                'eq_hist_grid_size': (random.randint(2, 20), random.randint(2, 20))
            }
            if type(augmentions) == tuple:
                # for augmention in augmentions:
                # if 'random' in augmentions[0]:
                augmentions[0](args_dict)
                augmentions[1](args_dict)
            else:
                augmentions(args_dict)
            # cv2.imshow('mask', self.mask)
        img = self.aug_img.get_image()
        if self.aug_mask is not None:
            mask = self.aug_mask.get_image()
        else:
            mask = None
        if np.any(img.shape == 0):
            img = np.zeros((100, 100), dtype=np.uint8)
        return img, mask


if __name__ == '__main__':
    img = cv2.imread('../data/green_screen/1/48.png')
    mask = cv2.imread('../data/mask/1/48.png')
    aug = AugmentImage(img, None)
    img, mask = aug.augment_image()
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    cv2.imshow('img_final', img)
    cv2.imshow('mask', mask)
    mask_img = cv2.bitwise_and(img, img, mask=mask[:,:,0])
    cv2.imshow('mask_img', mask_img)
    cv2.waitKey(0)