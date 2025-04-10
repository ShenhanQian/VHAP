# from https://github.com/tobias-kirschstein/nersemble-data/blob/f96aa8d9d482df53c40c51ecc07203646265e4f0/src/nersemble_data/util/color_correction.py

import colour
import numpy as np
from colour.characterisation import matrix_augmented_Cheung2004
from colour.utilities import as_float_array


def color_correction_Cheung2004_precomputed(
        image: np.ndarray,
        CCM: np.ndarray,
) -> np.ndarray:
    terms = CCM.shape[-1]
    RGB = as_float_array(image)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = matrix_augmented_Cheung2004(RGB, terms)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)


def correct_color(image: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    is_uint8 = image.dtype == np.uint8
    if is_uint8:
        image = image / 255.
    image_linear = colour.cctf_decoding(image)
    image_corrected = color_correction_Cheung2004_precomputed(image_linear, ccm)
    image_corrected = colour.cctf_encoding(image_corrected)
    if is_uint8:
        image_corrected = np.clip(image_corrected * 255, 0, 255).astype(np.uint8)

    return image_corrected
