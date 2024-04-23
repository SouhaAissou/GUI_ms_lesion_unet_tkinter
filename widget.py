from tkinter import *
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import os
import numpy as np
import cv2


def nifti_slice_to_png(nifti_file, output_dir, slice_index):
    # Load NIfTI image
    nifti_img = nib.load(nifti_file)
    img_data = nifti_img.get_fdata()

    # Get the base name of the NIfTI file
    base_name = os.path.basename(nifti_file).split('.')[0]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize pixel values to 0-255
    slice_data = img_data[:, :, slice_index]
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    slice_data = slice_data.astype(np.uint8)

    # Construct output file name
    slice_file = os.path.join(output_dir, f"{base_name}_slice_{slice_index}.png")
    plt.imsave(slice_file, slice_data, cmap='gray')

    print(f"Saved slice {slice_index} as {slice_file}")
    
    
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    # Convert the image to float32 (to avoid overflow during normalization)
    img_resized = img_resized.astype(np.float32)
    # Normalize the image
    img_resized = img_resized / 255.0
    # If you want to ensure the image is in the range [0, 1], you can use clip
    normalized_image = np.clip(normalized_image, 0, 1)
    return img_resized

# def browseFiles():
#     filename = filedialog.askopenfilename(initialdir = "/", 
#                                           title = "Select a File", 
#                                           filetypes = (("NIfTI files", "*.nii*"),
#                                                        ("GZipped NIfTI files", "*.nii.gz*"),
#                                                        ("all files", "*.*")))
#     # Change label contents
#     label_file.configure(text="File Opened: " + filename)

#     # Load the image
#     img = nib.load(filename)

#     # Get the image data
#     data = img.get_fdata()

#     # Load the model
#     model = tf.keras.models.load_model('C:\\Users\\dell\\Downloads\\10earlystoppingt2flair_27epoch\\kaggle\\working\\UNet_wavelet_fusion_150epoch_model_12.h5')

#     # Select a slice to segment
#     slice = data[:, :, data.shape[2] // 2]

#     # Preprocess the slice if necessary (e.g., resize, normalize, etc.)
#     # slice = preprocess(slice)
    

#     # Use the model to segment the slice
#     segmentation = model.predict(slice[None, ..., None])

#     # Display the original slice and the segmentation
#     fig, ax = plt.subplots(1, 2, figsize=[12, 6])
#     ax[0].imshow(slice.T, cmap="gray", origin="lower")
#     ax[1].imshow(segmentation[0, ..., 0].T, cmap="gray", origin="lower")

#     # Create a canvas and add it to the window
#     canvas = FigureCanvasTkAgg(fig, master=window)
#     canvas.draw()
#     canvas.get_tk_widget().pack()
#     label_file.configure(text="File Opened: " + filename)


# Initialize canvas as None outside the function
canvas = None
# Initialize slider as None outside the function
slider = None

def browseFiles():
    global canvas  # Use the global canvas variable
    global slider  # Use the global slider variable

    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("NIfTI files", "*.nii*"),
                                                       ("GZipped NIfTI files", "*.nii.gz*"),
                                                       ("all files", "*.*")))
    # Change label contents
    label_file.configure(text="File Opened: " + filename)

    # Load the image
    img = nib.load(filename)

    # Reorient the image
    img = nib.as_closest_canonical(img)

    # Get the image data
    data = img.get_fdata()

    # Create a subplot
    fig, ax = plt.subplots(figsize=[6, 6])

    # Plot the first slice
    slice = data[:, :, 0].T
    im = ax.imshow(slice, cmap="gray", origin="lower")

    # Clear the previous image from the canvas
    if canvas is not None:
        canvas.get_tk_widget().pack_forget()

    # Create a canvas and add it to the window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Remove the old slider
    if slider is not None:
        slider.pack_forget()

    # Create a slider
    slider = Scale(window, from_=0, to=data.shape[2] - 1, orient=HORIZONTAL, command=lambda s: update_slice(s, data, im))
    slider.pack()
    label_file.configure(text="File Opened: " + filename)
    
    

def update_slice(s, data, im):
    # Update the displayed slice
    slice = data[:, :, int(s)].T
    im.set_data(slice)
    plt.draw()

def update_slice(s, data, im):
    # Update the displayed slice
    slice = data[:, :, int(s)].T
    im.set_data(slice)
    plt.draw()
    
def on_closing():
    window.destroy()
    
                                                                                                
# Create the root window
window = Tk()

# Set window title
window.title('NIfTI Image Viewer')

# Set window size
window.geometry("500x500")

# Create a File Explorer label
main_label = Label(window, 
                            text="NIfTI Image Viewer",
                            font=("Arial", 16)
                            )
main_label.pack()

label_file = Label(window,
                    text = "Select a file",
                    font=("Arial", 12)
                    )
label_file.pack()

button_explore = Button(window, 
                        text="Browse Files",
                        command=browseFiles) 
button_explore.pack()

button_exit = Button(window, 
					text = "Exit",
					command = exit) 
button_exit.pack()

# Handle window close event
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()
