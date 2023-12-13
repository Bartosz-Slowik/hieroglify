# Standard library imports
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageGrab
from PIL import Image
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import match_template
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt



IMAGE_FILES = ["abcd.jpg","mno.jpg","cleopatra.jpg"]



def load_and_process_image(file_path, upscale_factor=2.0):
    # Read the image
    image = io.imread(file_path)
    # Upscale the image
    image = rescale(image, scale=upscale_factor, anti_aliasing=True, channel_axis=2)
    # Convert the image to grayscale and invert
    image = rgb2gray(image)
    image = 1 - image

    return image


def load_templates():
    templates = {
        'a': (load_and_process_image('./patterns/wzorzec_a.jpg'), 0.65),
        'b': (load_and_process_image('./patterns/wzorzec_b.jpg'), 0.8),
        'c': (load_and_process_image('./patterns/wzorzec_c.jpg'), 0.8),
        'd': (load_and_process_image('./patterns/wzorzec_d.jpg'), 0.85),
        'e': (load_and_process_image('./patterns/wzorzec_e.jpg'), 0.8),
        'f': (load_and_process_image('./patterns/wzorzec_f.jpg'), 0.8),
        'g': (load_and_process_image('./patterns/wzorzec_g.jpg'), 0.7),
        'h': (load_and_process_image('./patterns/wzorzec_h.jpg'), 0.8),
        'j': (load_and_process_image('./patterns/wzorzec_j.jpg'), 0.8),
        'l': (load_and_process_image('./patterns/wzorzec_l.jpg'), 0.8),
        'm': (load_and_process_image('./patterns/wzorzec_m.jpg'), 0.8),
        'n': (load_and_process_image('./patterns/wzorzec_n.jpg'), 0.75),
        'o': (load_and_process_image('./patterns/wzorzec_o.jpg'), 0.7),
        'p': (load_and_process_image('./patterns/wzorzec_p.jpg'), 0.8),
        'r': (load_and_process_image('./patterns/wzorzec_r.jpg'), 0.8),
        's': (load_and_process_image('./patterns/wzorzec_s.jpg'), 0.8),
        't': (load_and_process_image('./patterns/wzorzec_t.jpg'), 0.85),
        'w': (load_and_process_image('./patterns/wzorzec_w.jpg'), 0.8),
        'z': (load_and_process_image('./patterns/wzorzec_z.jpg'), 0.8),
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


def plot_detected_glyphs(image, found_glyphs, templates, word):
    # Create a new figure with a specific size (in inches)
    plt.figure(figsize=(10, 6))

    plt.imshow(image, cmap='gray')
    plt.axis('off')

    for i, (x, y, glyph_name, _) in enumerate(found_glyphs):
        template_height, template_width = templates[glyph_name][0].shape
        rect = plt.Rectangle((x, y), template_width, template_height, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    # Add the interpreted word to the plot
    plt.text(0, -20, f"Interpreted Word: {word}", fontsize=12)

    plt.show()


def process_image(image, templates):
    for key, (template, threshold) in templates.items():
        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            scaled_size = (min(template.shape[0], image.shape[0] - 1), min(template.shape[1], image.shape[1] - 1))
            templates[key] = (resize(template, scaled_size), threshold)

    found_glyphs = []
    for glyph, (template, threshold) in templates.items():
        found_glyphs.extend(detect_glyphs(template, image, threshold, glyph))

    ROW_HEIGHT = 166  # The height of each row in pixels

    found_glyphs.sort(key=lambda glyph: (glyph[1] // ROW_HEIGHT, glyph[0]))
    detected_letters = [glyph[2] for glyph in found_glyphs]
    word = ''.join(detected_letters)    
    print("Interpreted Word:", word)

    plot_detected_glyphs(image, found_glyphs, templates, word)



def main():
    templates = load_templates()

    # Create a new Tkinter window
    window = tk.Tk()
    window.geometry("300x500")  # Set the window size
    window.configure(bg='#FFF8E8')  # Set the background color

    window.title("Hieroglyphic Reader - Slowik, Samula")

    # Set the window icon
    window.iconbitmap('icon.ico')

    # Create a label at the top of the window inside a box
    label_frame = tk.LabelFrame(window, bg='#C1554D', font=('helvetica', 20))
    label_frame.pack(pady=10)  # Add padding
    label = tk.Label(label_frame, text="Hieroglyphic reader", bg='white', fg= "#C1554D", font=('helvetica', 20))
    label.pack()

    # Create a frame to hold the buttons
    frame = tk.Frame(window, bg='#FFF8E8')
    frame.pack(padx=10, pady=20)  # Add padding

    # Create a button that will open the file dialog
    open_file_button = tk.Button(frame, text="Open Image", command=lambda: open_image(templates))
    open_file_button.config(height=3, width=20, bg='#FFC75F', fg='black', font=('helvetica', 15))
    open_file_button.pack(pady=20)  # Add padding

    # Create a button that will get the image from the clipboard
    paste_image_button = tk.Button(frame, text="Paste Image", command=lambda: paste_image(templates))
    paste_image_button.config(height=3, width=20, bg='#FFC1B3', fg='black', font=('helvetica', 15))
    paste_image_button.pack(pady=20)  # Add padding

    # Create a button that will load example images
    examples_button = tk.Button(frame, text="Examples", command=lambda: load_examples(templates))
    examples_button.config(height=3, width=20, bg='#FE8A7F', fg='black', font=('helvetica', 15))
    examples_button.pack(pady=20)  # Add padding

    # Run the Tkinter event loop
    window.mainloop()


def load_examples(templates):
    # Get the list of example image files
    example_files = os.listdir('./examples')

    # Loop over the example files
    for filename in example_files:
        image = load_and_process_image("./examples/"+filename)
        process_image(image, templates)

def open_image(templates):
    # Open a file dialog and get the selected file path
    file_path = filedialog.askopenfilename()

    # Check if the selected file is a JPG image
    if not file_path.lower().endswith('.jpg'):
        messagebox.showerror("Error", "Selected file is not a JPG image.")
        return

    # Process the selected image
    image = load_and_process_image(file_path)
    process_image(image, templates)

def paste_image(templates):
    # Get the image from the clipboard
    image = get_image_from_clipboard()

    # Convert the PIL Image to a NumPy array and process it
    if image is not None:
        image = np.array(image)
        image = rescale(image, scale=2.0, anti_aliasing=True, channel_axis=2)
        # Convert the image to grayscale and invert
        image = rgb2gray(image)
        image = 1 - image
        process_image(image, templates)

def get_image_from_clipboard():
    # Get the image from the clipboard
    image = ImageGrab.grabclipboard()
    if image is None:
        print("No image data found on the clipboard.")
        return None

    # Convert the image to JPG and save it
    image = image.convert("RGB")
    image.save("clipboard_image.jpg", "JPEG")

    return image

if __name__ == "__main__":
    main()