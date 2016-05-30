# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt

# ---------------------------------------------- Variable Declarations -------------------------------------------------
direction_rows = 0
direction_columns = 1

scale_order = 2

noise_gaussian = "Gaussian Noise"
noise_salt_pepper = "Salt & Pepper Noise"

filter_haar = "Haar Filter"
filter_coiflet = "Coiflet Filter"

threshold_final = 1e5

threshold_original_haar = 15
threshold_original_coiflet = 14

threshold_gaussian_haar = 18
threshold_gaussian_coiflet = 16

threshold_snp_haar = 15
threshold_snp_coifflet = 13


# ------------------------------------------------- Defining Noises ----------------------------------------------------
def generateNoise(img_input, noise_type, level_val_1, level_val_2):
    if noise_type == noise_gaussian:
        height, width = img_input.shape

        mean = level_val_1
        var = level_val_2
        sigma = var ** 0.5  # Sample 20

        # Gaussian-distributed additive noise
        gauss = np.random.normal(mean, sigma, (height, width))
        gauss = gauss.reshape(height, width)
        img_gaussian_noise = img_input + gauss

        return img_gaussian_noise

    elif noise_type == noise_salt_pepper:
        img_salt_pepper_noise = img_input.copy()

        salt_vs_pepper = level_val_1
        amount = level_val_2

        # Salt mode : Replaces random pixels with 1
        num_salt = np.ceil(amount * img_input.size * salt_vs_pepper)
        pixels = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img_input.shape]
        img_salt_pepper_noise[pixels] = 1

        # Pepper mode : Replaces random pixels with 0
        num_pepper = np.ceil(amount * img_input.size * (1. - salt_vs_pepper))
        pixels = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img_input.shape]
        img_salt_pepper_noise[pixels] = 0

        return img_salt_pepper_noise


# --------------------------------------------- Defining Sampling Operation --------------------------------------------
def downSample(img_input, direction):
    if direction == direction_rows:
        h, w = img_input.shape
        img_output = np.zeros((h / 2, w))
        i = 0
        new = i
        while i < h:
            img_output[new, :] = img_input[i, :]
            new = new + 1
            i = i + 2

        return img_output

    elif direction == direction_columns:
        h, w = img_input.shape
        img_output = np.zeros((h, w / 2))
        i = 0
        new = i
        while i < w:
            img_output[:, new] = img_input[:, i]
            new = new + 1
            i = i + 2

        return img_output


def upSample(img_input, order):
    img_output = scipy.ndimage.zoom(img_input, order)
    return img_output


# ----------------------------------------- Implementation of Haar Wavelet Filter---------------------------------------
def haarConvolveRowsLowPass(img_input):
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("haarConvolveRowsLowPass : ")
    print(img_input_pad.shape)

    img_haar_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(0, width - 1):
            img_haar_convolved[x, y] = (img_input_pad[x, y] / math.sqrt(2)) + (
                img_input_pad[x, y + 1] / math.sqrt(2))

    return img_haar_convolved[0:height, 0:width - 1]


def haarConvolveColumnsLowPass(img_input):
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("haarConvolveColumnsLowPass : ")
    print(img_input_pad.shape)

    img_haar_convolved = np.zeros((height, width))

    for y in range(0, width):
        for x in range(0, height - 1):
            img_haar_convolved[x, y] = (img_input_pad[x, y] / math.sqrt(2)) + (
                img_input_pad[x + 1, y] / math.sqrt(2))

    return img_haar_convolved[0:height - 1, 0:width]


def haarConvolveRowsHighPass(img_input):
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("haarConvolveRowsHighPass : ")
    print(img_input_pad.shape)

    img_haar_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(0, width - 1):
            img_haar_convolved[x, y] = -(img_input_pad[x, y] / math.sqrt(2)) + (
                img_input_pad[x, y + 1] / math.sqrt(2))

    return img_haar_convolved[0:height, 0:width - 1]


def haarConvolveColumnsHighPass(img_input):
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("haarConvolveColumnsHighPass : ")
    print(img_input_pad.shape)

    img_haar_convolved = np.zeros((height, width))

    for y in range(0, width):
        for x in range(0, height - 1):
            img_haar_convolved[x, y] = -(img_input_pad[x, y] / math.sqrt(2)) + (
                img_input_pad[x + 1, y] / math.sqrt(2))

    return img_haar_convolved[0:height - 1, 0:width]


# ----------------------------------------- Implementation of Coiflet Wavelet Filter------------------------------------
def coifletConvolveRowsLowPass(img_input):
    # -0.0157   -0.0727    0.3849      0.8526    0.3379   -0.0727
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 2, 3, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("coifletConvolveRowsLowPass : ")
    print(img_input_pad.shape)

    img_coiflet_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(2, width - 3):
            img_coiflet_convolved[x, y] = img_input_pad[x, y - 2] * (-0.0157) + img_input_pad[x, y - 1] * (-0.0727) + \
                                          img_input_pad[x, y] * (0.3849) + img_input_pad[x, y + 1] * (0.8526) + \
                                          img_input_pad[x, y + 2] * (0.3379) + img_input_pad[x, y + 3] * (-0.0727)

    return img_coiflet_convolved[0:height, 2:width - 3]


def coifletConvolveColumnsLowPass(img_input):
    # -0.0157   -0.0727    0.3849      0.8526    0.3379   -0.0727
    img_input_pad = cv2.copyMakeBorder(img_input, 2, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("coifletConvolveColumnsLowPass : ")
    print(img_input_pad.shape)

    img_coiflet_convolved = np.zeros((height, width))

    for x in range(2, height - 3):
        for y in range(0, width):
            img_coiflet_convolved[x, y] = img_input_pad[x - 2, y] * (-0.0157) + img_input_pad[x - 1, y] * (-0.0727) + \
                                          img_input_pad[x, y] * (0.3849) + img_input_pad[x + 1, y] * (0.8526) + \
                                          img_input_pad[x + 2, y] * (0.3379) + img_input_pad[x + 3, y] * (-0.0727)

    return img_coiflet_convolved[2:height - 3, 0:width]


def coifletConvolveRowsHighPass(img_input):
    # 0.0727    0.3379   -0.8526    0.3849    0.0727   -0.0157; High
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 2, 3, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("coifletConvolveRowsHighPass : ")
    print(img_input_pad.shape)

    img_coiflet_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(2, width - 3):
            img_coiflet_convolved[x, y] = img_input_pad[x, y - 2] * (0.0727) + img_input_pad[x, y - 1] * (0.3379) + \
                                          img_input_pad[x, y] * (-0.8526) + img_input_pad[x, y + 1] * (0.3849) + \
                                          img_input_pad[x, y + 2] * (0.0727) + img_input_pad[x, y + 3] * (-0.0157)

    return img_coiflet_convolved[0:height, 2:width - 3]


def coifletConvolveColumnsHighPass(img_input):
    # 0.0727    0.3379   -0.8526    0.3849    0.0727   -0.0157; High
    img_input_pad = cv2.copyMakeBorder(img_input, 2, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape

    print("coifletConvolveColumnsHighPass : ")
    print(img_input_pad.shape)

    img_coiflet_convolved = np.zeros((height, width))

    for x in range(2, height - 3):
        for y in range(0, width):
            img_coiflet_convolved[x, y] = img_input_pad[x - 2, y] * (0.0727) + img_input_pad[x - 1, y] * (0.3379) + \
                                          img_input_pad[x, y] * (-0.8526) + img_input_pad[x + 1, y] * (0.3849) + \
                                          img_input_pad[x + 2, y] * (0.0727) + img_input_pad[x + 3, y] * (-0.0157)
    return img_coiflet_convolved[2:height - 3, 0:width]


# ----------------------------------- Implementation of Wavelet Filters (LL, LH, HL, HH) -------------------------------
def performLL(img_input, filter_wavelet):
    if filter_wavelet == filter_haar:
        img_L_rows = haarConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_L_columns = haarConvolveColumnsLowPass(img_L_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_L_rows = coifletConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_L_columns = coifletConvolveColumnsLowPass(img_L_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled


def performLH(img_input, filter_wavelet):
    if filter_wavelet == filter_haar:
        img_L_rows = haarConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_H_columns = haarConvolveColumnsHighPass(img_L_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_L_rows = coifletConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_H_columns = coifletConvolveColumnsHighPass(img_L_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled


def performHL(img_input, filter_wavelet):
    if filter_wavelet == filter_haar:
        img_H_rows = haarConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_L_columns = haarConvolveColumnsLowPass(img_H_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_H_rows = coifletConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_L_columns = coifletConvolveColumnsLowPass(img_H_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled


def performHH(img_input, filter_wavelet):
    if filter_wavelet == filter_haar:
        img_H_rows = haarConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_H_columns = haarConvolveColumnsHighPass(img_H_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_H_rows = coifletConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_H_columns = coifletConvolveColumnsHighPass(img_H_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled


# --------------------------------------------- Pipeline for Wavelet Edge Detection-------------------------------------
def pipeline(img_input, filter_wavelet, noise, threshold_subband):
    # ------------------------ Step 1 : Applying Wavelet Filter, Threshold and Down Sampling ---------------------------
    # Level 1 : Perform filters LL, LH, HL and HH on Input Image of size [M, N] and generate [M/2, N/2] size Image
    img_LL_2 = performLL(img_input, filter_wavelet)
    img_LH_2 = performLH(img_input, filter_wavelet)
    img_HL_2 = performHL(img_input, filter_wavelet)
    img_HH_2 = performHH(img_input, filter_wavelet)

    printIntermediateResults(img_LL_2, img_LH_2, img_HL_2, img_HH_2, filter_wavelet, noise, "[M/2, N/2] Before Threshold")

    # Applying threshold to LH, HL and HH subbands
    img_LH_2 = performThresholding(img_LH_2, threshold_subband, False)
    img_HL_2 = performThresholding(img_HL_2, threshold_subband, False)
    img_HH_2 = performThresholding(img_HH_2, threshold_subband, False)

    printIntermediateResults(img_LL_2, img_LH_2, img_HL_2, img_HH_2, filter_wavelet, noise, "[M/2, N/2] After Threshold")

    # Level 2 : Perform filters LL, LH, HL and HH on LL Image of size [M/2, N/2] and generate [M/4, N/4] size Image
    img_LL_4 = performLL(img_LL_2, filter_wavelet)
    img_LH_4 = performLH(img_LL_2, filter_wavelet)
    img_HL_4 = performHL(img_LL_2, filter_wavelet)
    img_HH_4 = performHH(img_LL_2, filter_wavelet)

    # printIntermediateResults(img_LL_4, img_LH_4, img_HL_4, img_HH_4, filter_wavelet, noise, "[M/4, N/4] Before Threshold")

    # Applying threshold to LH, HL and HH subbands
    img_LH_4 = performThresholding(img_LH_4, threshold_subband, False)
    img_HL_4 = performThresholding(img_HL_4, threshold_subband, False)
    img_HH_4 = performThresholding(img_HH_4, threshold_subband, False)

    # printIntermediateResults(img_LL_4, img_LH_4, img_HL_4, img_HH_4, filter_wavelet, noise, "[M/4, N/4] After Threshold")

    # Level 3 : Perform filters LL, LH, HL and HH on LL Image of size [M/4, N/4] and generate [M/8, N/8] size Image
    img_LL_8 = performLL(img_LL_4, filter_wavelet)
    img_LH_8 = performLH(img_LL_4, filter_wavelet)
    img_HL_8 = performHL(img_LL_4, filter_wavelet)
    img_HH_8 = performHH(img_LL_4, filter_wavelet)

    # printIntermediateResults(img_LL_8, img_LH_8, img_HL_8, img_HH_8, filter_wavelet, noise,"[M/8, N/8] Before Threshold")

    # Applying threshold to LH, HL and HH subbands
    img_LH_8 = performThresholding(img_LH_8, threshold_subband, False)
    img_HL_8 = performThresholding(img_HL_8, threshold_subband, False)
    img_HH_8 = performThresholding(img_HH_8, threshold_subband, False)

    # printIntermediateResults(img_LL_8, img_LH_8, img_HL_8, img_HH_8, filter_wavelet, noise, "[M/8, N/8] After Threshold")

    # Level 4 : Perform filters LL, LH, HL and HH on LL Image of size [M/8, N/8] and generate [M/16, N/16] size Image
    img_LL_16 = performLL(img_LL_8, filter_wavelet)
    img_LH_16 = performLH(img_LL_8, filter_wavelet)
    img_HL_16 = performHL(img_LL_8, filter_wavelet)
    img_HH_16 = performHH(img_LL_8, filter_wavelet)

    # printIntermediateResults(img_LL_16, img_LH_16, img_HL_16, img_HH_16, filter_wavelet, noise, "[M/16, N/16] Before Threshold")

    # Applying threshold to LH, HL and HH subbands
    img_LH_16 = performThresholding(img_LH_16, threshold_subband, False)
    img_HL_16 = performThresholding(img_HL_16, threshold_subband, False)
    img_HH_16 = performThresholding(img_HH_16, threshold_subband, False)

    # printIntermediateResults(img_LL_16, img_LH_16, img_HL_16, img_HH_16, filter_wavelet, noise, "[M/16, N/16] After Threshold")

    # --------------------------------- Step 2 : Up Sampling and Image Matrix Multiplication ---------------------------
    # Blowing Up Level 4 to Level 3 : Up Sampling LH, HL and HH Images of size [M/16, N/16] to size [M/8, N/8]
    img_LH_16_blown = upSample(img_LH_16, scale_order)
    img_HL_16_blown = upSample(img_HL_16, scale_order)
    img_HH_16_blown = upSample(img_HH_16, scale_order)

    # printIntermediateResults(img_LL_16, img_LH_16_blown, img_HL_16_blown, img_HH_16_blown, filter_wavelet, noise, "[M/16, N/16] Blown to [M/8, N/8]")

    # Element wise Multiplication Level 4 blown up Images of size [M/8, N/8] with original Level 3 Images of size [M/8, N/8]
    img_LH_8_new = img_LH_16_blown * img_LH_8
    img_HL_8_new = img_HL_16_blown * img_HL_8
    img_HH_8_new = img_HH_16_blown * img_HH_8

    # printIntermediateResults(img_LL_8, img_LH_8_new, img_HL_8_new, img_HH_8_new, filter_wavelet, noise, "[M/8, N/8] After Multiplying")

    # Blowing Up Level 3 to Level 2 : Up Sampling LH, HL and HH Images of size [M/8, N/8] to size [M/4, N/4]
    img_LH_8_blown = upSample(img_LH_8_new, scale_order)
    img_HL_8_blown = upSample(img_HL_8_new, scale_order)
    img_HH_8_blown = upSample(img_HH_8_new, scale_order)

    # printIntermediateResults(img_LL_8, img_LH_8_blown, img_HL_8_blown, img_HH_8_blown, filter_wavelet, noise, "[M/8, N/8] Blown to [M/4, N/4]")

    # Element wise Multiplication Level 3 blown up Images of size [M/4, N/4] with original Level 2 Images of size [M/4, N/4]
    img_LH_4_new = img_LH_8_blown * img_LH_4
    img_HL_4_new = img_HL_8_blown * img_HL_4
    img_HH_4_new = img_HH_8_blown * img_HH_4

    # printIntermediateResults(img_LL_4, img_LH_4_new, img_HL_4_new, img_HH_4_new, filter_wavelet, noise, "[M/4, N/4] After Multiplying")

    # Blowing Up Level 2 to Level 1 : Up Sampling LH, HL and HH Images of size [M/4, N/4] to size [M/2, N/2]
    img_LH_4_blown = upSample(img_LH_4_new, scale_order)
    img_HL_4_blown = upSample(img_HL_4_new, scale_order)
    img_HH_4_blown = upSample(img_HH_4_new, scale_order)

    # printIntermediateResults(img_LL_4, img_LH_4_blown, img_HL_4_blown, img_HH_4_blown, filter_wavelet, noise, "[M/4, N/4] Blown to [M/2, N/2]")

    # Element wise Multiplication Level 2 blown up Images of size [M/2, N/2] with original Level 1 Images of size [M/2, N/2]
    img_LH_2_new = img_LH_4_blown * img_LH_2
    img_HL_2_new = img_HL_4_blown * img_HL_2
    img_HH_2_new = img_HH_4_blown * img_HH_2

    # printIntermediateResults(img_LL_2, img_LH_2_new, img_HL_2_new, img_HH_2_new, filter_wavelet, noise, "[M/2, N/2] After Multiplying")

    print img_LH_2_new.shape

    # --------------------------------------- Step 3 : Generating Final Edge map ---------------------------------------
    img_LH_final = np.multiply(img_LH_2_new, img_LH_2_new)
    img_HL_final = np.multiply(img_HL_2_new, img_HL_2_new)
    img_HH_final = np.multiply(img_HH_2_new, img_HH_2_new)
    img_final_edges = np.sqrt(img_LH_final + img_HL_final + img_HH_final)

    return True, img_final_edges


# ------------------------------------------ Print Intermediate Results ------------------------------------------------
def printIntermediateResults(img_LL, img_LH, img_HL, img_HH, filter_wavelet, noise, msg):
    title_pipeline = "LL           Intermediate %s with %s and %s" % (msg, noise, filter_wavelet)

    plt.subplot(221), plt.imshow(img_LL, cmap='gray'), plt.title(title_pipeline, loc='left')
    plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(img_LH, cmap='gray'), plt.title('LH', loc='right')
    plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(img_HL, cmap='gray'), plt.title('HL')
    plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(img_HH, cmap='gray'), plt.title('HH')
    plt.xticks([]), plt.yticks([])

    plt.show()


# ---------------------------------------------- Applying Threshold -------------------------------------------------
def performThresholding(img_input, val_threshold, isFinalImg):
    low_values_indices = img_input < val_threshold  # Where values are low
    img_input[low_values_indices] = 0  # All low values set to 0

    if isFinalImg:
        low_values_indices = img_input > 0  # Where values are low
        img_input[low_values_indices] = 255  # All low values set to 0

    return img_input


# ----------------------------------------- Printing Final Comparison Results ------------------------------------------
def printFinalComparisonResults(img_orig_haar_edges, img_orig_coiflet_edges, img_gauss_haar_edges,
                                img_gauss_coiflet_edges, img_sp_haar_edges, img_sp_coiflet_edges):
    img_orig_haar_edges = performThresholding(img_orig_haar_edges, threshold_final, True)
    img_orig_coiflet_edges = performThresholding(img_orig_coiflet_edges, threshold_final, True)

    img_gauss_haar_edges = performThresholding(img_gauss_haar_edges, threshold_final, True)
    img_gauss_coiflet_edges = performThresholding(img_gauss_coiflet_edges, threshold_final, True)

    img_sp_haar_edges = performThresholding(img_sp_haar_edges, threshold_final, True)
    img_sp_coiflet_edges = performThresholding(img_sp_coiflet_edges, threshold_final, True)

    plt.subplot(231), plt.imshow(img_orig_haar_edges, cmap='gray'), plt.title('Original Image with Haar Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(img_gauss_haar_edges, cmap='gray'), plt.title('Gaussian Noise and Haar Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(img_sp_haar_edges, cmap='gray'), plt.title('Salt Paper Noise and Haar Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(img_orig_coiflet_edges, cmap='gray'), plt.title('Original Image with Coiflet Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(img_gauss_coiflet_edges, cmap='gray'), plt.title('Gaussian Noise and Coiflet Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(236), plt.imshow(img_sp_coiflet_edges, cmap='gray'), plt.title('Salt Paper Noise and Coiflet Filter')
    plt.xticks([]), plt.yticks([])

    plt.show()


# ----------------------------------------------- Program Main Function ------------------------------------------------
if __name__ == "__main__":
    Images = ["Lena.jpg", "Peppers.jpg", "Carriage.jpg"]
    Noises = [noise_gaussian, noise_salt_pepper]
    Filters = [filter_haar, filter_coiflet]

    # [Mean, Variance]
    Noise_Levels_Gaussian = [[20, 40]]

    # [salt_vs_pepper, amount of noise]
    Noise_Levels_Salt_Pepper = [[0.5, 0.003]]

    for image in Images:
        img_original = cv2.imread(image)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        img_original_haar_edges = img_original
        img_original_coiflet_edges = img_original

        img_gaussian_haar_edges = img_original
        img_gaussian_coiflet_edges = img_original

        img_salt_pepper_haar_edges = img_original
        img_salt_pepper_coiflet_edges = img_original

        for noise in Noises:
            if noise == noise_gaussian:
                for noise_level in Noise_Levels_Gaussian:
                    img_noisy_gaussian = generateNoise(img_original, noise, noise_level[0], noise_level[1])

                    plt.subplot(121), plt.imshow(img_original, cmap='gray'), plt.title('Original Image')
                    plt.xticks([]), plt.yticks([])

                    title = "%s (Mean: %s, Variance: %s)" % (
                        noise, "{0:.2f}".format(noise_level[0]), "{0:.2f}".format(noise_level[1]))
                    plt.subplot(122), plt.imshow(img_noisy_gaussian, cmap='gray'), plt.title(title)
                    plt.xticks([]), plt.yticks([])

                    plt.show()

                    for filter_wavelet in Filters:
                        if filter_wavelet == filter_haar:
                            success, img_original_haar_edges = pipeline(img_original, filter_wavelet, 'Original Image',
                                                                        threshold_original_haar)
                            success, img_gaussian_haar_edges = pipeline(img_noisy_gaussian, filter_wavelet, noise,
                                                                        threshold_gaussian_haar)

                        elif filter_wavelet == filter_coiflet:
                            success, img_original_coiflet_edges = pipeline(img_original, filter_wavelet,
                                                                           'Original Image',
                                                                           threshold_original_coiflet)
                            success, img_gaussian_coiflet_edges = pipeline(img_noisy_gaussian, filter_wavelet, noise,
                                                                           threshold_gaussian_coiflet)

                        if success:
                            print(
                                'Wavelet Edge Detection Pipeline for ' + image + ' with ' + noise + ' and ' + filter_wavelet + ' executed successfully')

            elif noise == noise_salt_pepper:
                for noise_level in Noise_Levels_Salt_Pepper:
                    img_noisy_salt_pepper = generateNoise(img_original, noise, noise_level[0], noise_level[1])

                    plt.subplot(121), plt.imshow(img_original, cmap='gray'), plt.title('Original Image')
                    plt.xticks([]), plt.yticks([])

                    title = "%s (Salt vs Pepper: %s, Amount: %s)" % (
                        noise, "{0:.2f}".format(noise_level[0]), "{0:.3f}".format(noise_level[1]))
                    plt.subplot(122), plt.imshow(img_noisy_salt_pepper, cmap='gray'), plt.title(title)
                    plt.xticks([]), plt.yticks([])

                    plt.show()

                    for filter_wavelet in Filters:
                        if filter_wavelet == filter_haar:
                            success, img_salt_pepper_haar_edges = pipeline(img_noisy_salt_pepper, filter_wavelet, noise,
                                                                           threshold_snp_haar)

                        elif filter_wavelet == filter_coiflet:
                            success, img_salt_pepper_coiflet_edges = pipeline(img_noisy_salt_pepper, filter_wavelet,
                                                                              noise, threshold_snp_coifflet)

                        if success:
                            print(
                                'Wavelet Edge Detection Pipeline for ' + image + ' with ' + noise + ' and ' + filter_wavelet + ' executed successfully')

        printFinalComparisonResults(img_original_haar_edges, img_original_coiflet_edges, img_gaussian_haar_edges,
                                    img_gaussian_coiflet_edges, img_salt_pepper_haar_edges,
                                    img_salt_pepper_coiflet_edges)
