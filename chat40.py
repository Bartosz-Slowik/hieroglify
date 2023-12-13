#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Improved Script for Hieroglyph Detection and Interpretation

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage.color import rgb2gray
from skimage import io
from skimage.transform import resize

def load_and_process_image(file_path):
        # Read and convert the image to grayscale
        image = rgb2gray(io.imread(file_path))
        
        # Invert the image: subtract it from 1 for floating-point data
        return 1 - image
    
def detect_glyphs(template, image, tolerance, glyph_name):
    if image is None:
        return

    result = match_template(image, template)
    coordinates = []

    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    while result[y, x] > tolerance:
        coordinates.append((x, y, glyph_name, result[y, x]))
        
        # Define the size of the area to zero out
        size = 20  # Adjust this value as needed

        # Zero out a region around the maximum
        result[max(0, y - size):min(result.shape[0], y + size + 1),
               max(0, x - size):min(result.shape[1], x + size + 1)] = 0

        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]

    return coordinates




# Load Templates
templates = {
    'u': load_and_process_image('./patterns/wzorzec_u.JPG'),
    'kh': load_and_process_image('./patterns/wzorzec_kh.JPG'),
    'f': load_and_process_image('./patterns/wzorzec_f.JPG'),
    'a': load_and_process_image('./patterns/wzorzec_a.JPG'),
    'i': load_and_process_image('./patterns/wzorzec_i.JPG'),
    'k': load_and_process_image('./patterns/wzorzec_k.JPG'),
    'l': load_and_process_image('./patterns/wzorzec_l.JPG'),
    'm': load_and_process_image('./patterns/wzorzec_m.JPG'),
    'o': load_and_process_image('./patterns/wzorzec_o.JPG'),
    'p': load_and_process_image('./patterns/wzorzec_p.JPG'),
    'r': load_and_process_image('./patterns/wzorzec_r.JPG'),
    's': load_and_process_image('./patterns/wzorzec_s.JPG'),
    't': load_and_process_image('./patterns/wzorzec_t.JPG'),
    # Add other templates as needed
}
egyptian_to_polish = {'khufu': "Cheops", 'ptolmiis': "Ptolemeusz", 'kliopatra': "Kleopatra"}

def process_image(image):
    for key, template in templates.items():
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            print(f"Resizing template {key} to fit the image.")
            scaled_size = (min(template.shape[0], image.shape[0] - 1), min(template.shape[1], image.shape[1] - 1))
            templates[key] = resize(template, scaled_size)

    # Main Processing
    found_glyphs = []
    for glyph, template in templates.items():
        found_glyphs.extend(detect_glyphs(template, image, 0.75, glyph))

    # Sorting and interpretation
    found_glyphs.sort(key=lambda x: x[0])  # Sort by x-coordinate
    print("Found glyphs:", found_glyphs)
    detected_letters = [glyph[2] for glyph in found_glyphs]
    word = ''.join(detected_letters)

    print("\nDetected Egyptian Letters:", detected_letters)
    print("Interpreted Word:", word)
    print("Translation:", egyptian_to_polish.get(word, "Unknown"))


# Load Images
image_khufu = load_and_process_image('chufu.jpg')
image_kleopatra = load_and_process_image('kleopatra.jpg')
image_klipt = load_and_process_image('cleo2.jpg')
process_image(image_khufu)
process_image(image_kleopatra)
process_image(image_klipt)
