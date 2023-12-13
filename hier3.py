#!/usr/bin/env python
# -*- coding: utf-8 -*-
#https://www.egyptianhieroglyphs.net/egyptian-hieroglyphs/lesson-1/


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage.color import rgb2gray
from skimage import io
from skimage.transform import resize
import os
from skimage.transform import rescale

# Define constants
TEMPLATES_FOLDER = './patterns/'
IMAGE_FILES = ['chufu.jpg', 'kleopatra.jpg', 'cleo2.jpg']
THRESHOLD = 0.71
SIZE_TO_ZERO_OUT = 20
egyptian_to_polish = {'khufu': "Cheops", 'ptolmiis': "Ptolemeusz", 'kliopatra': "Kleopatra"}



def load_and_process_image(file_path, upscale_factor=2.0):
    # Read the image
    image = io.imread(file_path)

    # Upscale the image
    image = rescale(image, scale=upscale_factor, anti_aliasing=True, multichannel=True)

    # Convert the image to grayscale and invert
    image = rgb2gray(image)
    image = 1 - image

    return image

def load_templates():
    templates = {
        'u': (load_and_process_image('./patterns/wzorzec_u.JPG'), 0.71),
        'kh': (load_and_process_image('./patterns/wzorzec_kh.JPG'), 0.72),
        'f': (load_and_process_image('./patterns/wzorzec_f.JPG'), 0.73),
        'a': (load_and_process_image('./patterns/wzorzec_a.JPG'), 0.38),
        'i': (load_and_process_image('./patterns/wzorzec_i.JPG'), 0.75),
        'k': (load_and_process_image('./patterns/wzorzec_k.JPG'), 0.76),
        'l': (load_and_process_image('./patterns/wzorzec_l.JPG'), 0.6),
        'm': (load_and_process_image('./patterns/wzorzec_m.JPG'), 0.78),
        'o': (load_and_process_image('./patterns/wzorzec_o.JPG'), 0.4),
        'p': (load_and_process_image('./patterns/wzorzec_p.JPG'), 0.7),
        'r': (load_and_process_image('./patterns/wzorzec_r.JPG'), 0.52),
        's': (load_and_process_image('./patterns/wzorzec_s.JPG'), 0.82),
        't': (load_and_process_image('./patterns/wzorzec_t.JPG'), 0.83),
        # Add other templates as needed
    }

    return templates

def detect_glyphs(template, image, tolerance, glyph_name):
    if image is None:
        return

    result = match_template(image, template)
    coordinates = []

    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    while result[y, x] > tolerance:
        coordinates.append((x, y, glyph_name, result[y, x]))

        # Define the size of the area to zero out based on a slightly smaller fraction of the detected letter size
        size = max(template.shape) // 4  # Adjust as needed for zeroing out
        zero_out_size = int(size * 0.8)  # Reduce the zero-out area by 20%

        # Zero out a region around the maximum
        result[max(0, y - zero_out_size):min(result.shape[0], y + zero_out_size + 1),
               max(0, x - zero_out_size):min(result.shape[1], x + zero_out_size + 1)] = 0

        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]

    return coordinates


def plot_detected_glyphs(image, found_glyphs, templates):
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    for i, (x, y, glyph_name, _) in enumerate(found_glyphs):
        template_height, template_width = templates[glyph_name][0].shape

        # Calculate the rectangle coordinates around the detected glyph based on template dimensions
        rect_x = x
        rect_y = y
        rect_width = template_width
        rect_height = template_height

        rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()


def process_image(image, templates):
    for key, (template, threshold) in templates.items():
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            scaled_size = (min(template.shape[0], image.shape[0] - 1), min(template.shape[1], image.shape[1] - 1))
            templates[key] = (resize(template, scaled_size), threshold)

    found_glyphs = []
    for glyph, (template, threshold) in templates.items():
        found_glyphs.extend(detect_glyphs(template, image, threshold, glyph))

    found_glyphs.sort(key=lambda x: x[0])  # Sort by x-coordinate
    detected_letters = [glyph[2] for glyph in found_glyphs]
    word = ''.join(detected_letters)    
    print("Interpreted Word:", word)
    print("Translation:", egyptian_to_polish.get(word, "Unknown"))

    plot_detected_glyphs(image, found_glyphs, templates)

def main():
    templates = load_templates()

    for image_file in IMAGE_FILES:
        image = load_and_process_image(image_file)
        process_image(image, templates)

if __name__ == "__main__":
    main()
