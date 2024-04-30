from tkinter import *
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import os
import numpy as np
import cv2
import h5py
import ttkbootstrap as ttk

from PIL import Image
import io
from scipy.ndimage import zoom


   
    
def nifti_slice_to_png(nifti_file, slice_index):
    nifti_img = nib.load(nifti_file)
    img_data = nifti_img.get_fdata()

    # Normalize pixel values to 0-255
    slice_data = img_data[:, :, slice_index]
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-7) * 255
    slice_data = slice_data.astype(np.uint8)
    img = Image.fromarray(slice_data)

    # Save PIL Image to BytesIO object in PNG format
    png_image = io.BytesIO()
    img.save(png_image, format='PNG')

    # Reset BytesIO object's position to the beginning
    png_image.seek(0)
    return png_image

def read_image(png_image):
    # Read the image from the BytesIO object
    img = Image.open(png_image)
    # Convert the image to grayscale and resize it
    img = img.convert('L').resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.clip(img_array, 0, 1)
    return img_array


def load_h5_model(file_path):
    model = tf.keras.models.load_model(file_path)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


canvas = None
slider = None
button_segment = None

def browseFiles():
    global canvas  
    global slider
    global button_segment

    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("NIfTI files", "*.nii*"),
                                                       ("GZipped NIfTI files", "*.nii.gz*"),
                                                       ("all files", "*.*")))
   
    label_file.configure(text="File Opened: " + filename)
    
    def segment_slice():
        model = load_h5_model('C:\\Users\\dell\\Downloads\\10earlystoppingt2flair_27epoch\\kaggle\\working\\UNet_wavelet_fusion_150epoch_model_12.h5')
        
        # Select the displayed slice to segment
        slice_index = slider.get()
        # Get the original slice size
        original_size = slice.shape
        slice_to_segment = read_image(nifti_slice_to_png(filename, slice_index))
        slice_to_segment = np.expand_dims(slice_to_segment, axis=-1)

        # Use the model to segment the slice
        segmentation = model.predict(slice_to_segment[None, ..., None])
        
         # Resize the segmentation to the original slice size
        resized_segmentation = zoom(segmentation[0, :, :, 0], (original_size[0] / segmentation.shape[1], original_size[1] / segmentation.shape[2]))


        # Display the original slice and the segmentation
        fig, ax = plt.subplots( figsize=[12, 6])
        # ax[0].imshow(slice_to_segment, cmap="gray", origin="lower")
        ax.imshow(resized_segmentation, cmap="gray", origin="lower")
        
        # Create a canvas and add it to the window
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=RIGHT)

    # Load the image
    img = nib.load(filename)

    # Get the image data
    data = img.get_fdata()
    
    print('Shape:', data.shape)
    print('Max:', data.max())
    print('Min:', data.min())
    
    # Remove the old slider
    if slider is not None:
        slider.pack_forget()

    # Create a slider
    slider = Scale(window, from_=0, to=data.shape[2] - 1, orient=HORIZONTAL, command=lambda s: update_slice(s, data, im))
    slider.pack()
    
    # Remove the old segment button
    if button_segment is not None:
        button_segment.pack_forget()
    
    button_segment = Button(window, 
                        text="segment this slice",
                        command=segment_slice) 
    button_segment.pack()

    # Create a subplot
    fig, ax = plt.subplots(figsize=[6, 6])

    # Plot the first slice
    slice = data[:, :, 0]
    im = ax.imshow(slice, cmap="gray", origin="lower")

    # Clear the previous image from the canvas
    if canvas is not None:
        canvas.get_tk_widget().pack_forget()

    # Create a canvas and add it to the window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    # display the canvas on the left side of the window
    canvas.get_tk_widget().pack(side=LEFT)

    
    

def update_slice(s, data, im):
    # Update the displayed slice
    slice = data[:, :,int(s)]
    im.set_data(slice)
    plt.draw()


    
def on_closing():
    window.destroy()
    
                                                                                                
# Create the root window
window = Tk()

# Set window title
window.title('NIfTI Image Viewer')

# Set window size to full screen
window.state('zoomed')
window.config(bg="skyblue")

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
window.protocol("WM_DELETE_WINDOW", exit)
window.mainloop()
