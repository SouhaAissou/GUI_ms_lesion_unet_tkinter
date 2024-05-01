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
from scipy.ndimage import label
import sv_ttk



# ======================================================================================================================
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
# ======================================================================================================================
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
# ======================================================================================================================
def load_h5_model(file_path):
    model = tf.keras.models.load_model(file_path)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
# ======================================================================================================================
canvas = None
slider = None
button_segment = None
segmented_canvas = None
button_overlay = None


    
def browseFiles():
    global canvas  
    global slider
    global button_segment
    global segmented_canvas
    global button_overlay

    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("NIfTI files", "*.nii*"),
                                                       ("GZipped NIfTI files", "*.nii.gz*"),
                                                       ("all files", "*.*")))
   
    label_file.configure(text="File Opened: " + filename)
    
    if canvas is not None:
        canvas.get_tk_widget().destroy()

    if segmented_canvas is not None:
        segmented_canvas.get_tk_widget().destroy()
    
    # --------------------------------------------------------------------------------------------
    def segment_slice():
        global segmented_canvas
        global button_overlay
            

        model = load_h5_model('UNet_wavelet_fusion_150epoch_model_12.h5')
        
        slice_index = slider.get()
        original_size = slice.shape
        slice_to_segment = read_image(nifti_slice_to_png(filename, slice_index))
        slice_to_segment = np.expand_dims(slice_to_segment, axis=-1)

        segmentation = model.predict(slice_to_segment[None, ..., None])
        resized_segmentation = zoom(segmentation[0, :, :, 0], (original_size[0] / segmentation.shape[1], original_size[1] / segmentation.shape[2]))

        threshold = 0.5  
        resized_segmentation = (resized_segmentation > threshold).astype(int)

        labeled_array, num_features = label(resized_segmentation)
        print(f"Number of lesions: {num_features}")
        print(f"Lesion sizes: {np.bincount(labeled_array.flat)}")

        frame_width = segmented_image_frame.winfo_width()
        frame_height = segmented_image_frame.winfo_height()

        dpi = min(frame_width / 4, frame_height / 4)
        
        fig, ax = plt.subplots( figsize=[4, 4], dpi=dpi)
        ax.imshow(resized_segmentation, cmap="gray", origin="lower")
        ax.axis('off')
        
        if segmented_canvas is not None:
            segmented_canvas.get_tk_widget().destroy()

        segmented_canvas = FigureCanvasTkAgg(fig, master=segmented_image_frame)
        segmented_canvas.draw()
        segmented_canvas.get_tk_widget().grid(sticky='nsew')
        number_of_lesions = num_features
        number_of_lesions_label = Label(actions_frame, text=f"Number of lesions: {number_of_lesions}")
        number_of_lesions_label.grid(row=4, column=0, sticky='n')
        # --------------------------------------------------------------------------------------------
        def overlay_images():
            new_window = Toplevel(window)
            new_window.title("Overlay Image")

            fig, ax = plt.subplots(figsize=[6, 6])
            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

            original_slice = data[:, :, slice_index]
            ax.imshow(original_slice, cmap='gray', origin='lower')
            ax.imshow(resized_segmentation, cmap='Reds', alpha=0.3, origin='lower')
            ax.axis("off")
            canvas.draw()
        # --------------------------------------------------------------------------------------------
        if button_overlay is None:
            button_overlay = Button(actions_frame, 
                                text="Overlay Images",
                                command=overlay_images) 
            button_overlay.grid(row=5, column=0, sticky='n')
        else:
            button_overlay.configure(command=overlay_images)
    # --------------------------------------------------------------------------------------------
            
    # Load the image
    img = nib.load(filename)
    data = img.get_fdata()
    
    print('Shape:', data.shape)
    print('Max:', data.max())
    print('Min:', data.min())
    
    # Remove the old slider
    if slider is not None:
        slider.pack_forget()

    # Create a slider
    slider = Scale(actions_frame, from_=0, to=data.shape[2] - 1, orient=HORIZONTAL, command=lambda s: update_slice(s, data, im))
    slider.grid(row=2, column=0, sticky='n')
    
    # Remove the old segment button
    if button_segment is not None:
        button_segment.pack_forget()
    
    button_segment = Button(actions_frame, 
                        text="segment this slice",
                        command=segment_slice) 
    button_segment.grid(row=3, column=0, sticky='n')
    
    
    # Get the dimensions of original_image_frame
    frame_width = original_image_frame.winfo_width()
    frame_height = original_image_frame.winfo_height()

    # Calculate the required dpi for the figure
    dpi = min(frame_width / 4, frame_height / 4)

    # Display the original image
    fig, ax = plt.subplots(figsize=[4, 4], dpi=dpi)
    ax.axis('off')

    # Plot the first slice
    slice = data[:, :, 0]
    im = ax.imshow(slice, cmap="gray", origin="lower")
    ax.axis('off')

    # Create a canvas and add it to the window
    canvas = FigureCanvasTkAgg(fig, master=original_image_frame)
    canvas.draw()
    # display the canvas on the left side of the window
    canvas.get_tk_widget().grid(sticky='nsew')
    
# ======================================================================================================================


def update_slice(s, data, im):
    # Update the displayed slice
    # slice = data[:, :, int(s)]
    # im.set_data(slice)
    # plt.draw()
    slice_index = int(s)
    slice_data = data[:, :, slice_index]
    im.set_data(slice_data)
    canvas.draw() 

# ======================================================================================================================



# ======================================================================================================================
# Create the window an layout
# ======================================================================================================================

window = Tk()

window.title("MS leisons Segmentation")
window.state('zoomed')

style = ttk.Style()
style.configure("Custom.TFrame", background='#c4d7f8')

window.rowconfigure(0, weight=3)
window.rowconfigure(1, weight=20)
window.columnconfigure(0, weight=3)
window.columnconfigure(1, weight=8)
window.columnconfigure(2, weight=8)

window.configure(bg='#c4d7f8') 

main_label_frame = ttk.Frame(window, relief='solid', borderwidth=1)
main_label_frame.grid(row=0, column=0, columnspan=3, sticky='nsew')
main_label = Label(main_label_frame, text="MS Leison segmenation", font=("Arial", 16, "bold"))
main_label.pack(fill='both', expand=True)

actions_frame = ttk.Frame(window, relief='solid', borderwidth=1)
actions_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)

original_image_frame = ttk.Frame(window, relief='solid', borderwidth=1)
original_image_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)
original_image_frame.grid_propagate(False)

segmented_image_frame = ttk.Frame(window, relief='solid', borderwidth=1)
segmented_image_frame.grid(row=1, column=2, sticky='nsew', padx=10, pady=10)
segmented_image_frame.grid_propagate(False)


actions_frame.rowconfigure(0, weight=1)
actions_frame.rowconfigure(1, weight=1)
actions_frame.rowconfigure(2, weight=1)
actions_frame.rowconfigure(3, weight=1)
actions_frame.rowconfigure(4, weight=1)
actions_frame.rowconfigure(5, weight=1)
actions_frame.rowconfigure(5, weight=10)


actions_frame.grid_propagate(False)

label_file = Label(actions_frame,
                    text = "Select a file",
                    font=("Arial", 12),
                    wraplength=200
                    )
label_file.grid(row=0, column=0, sticky='n')

button_explore = Button(actions_frame, 
                        text="Browse Files",
                        command=browseFiles) 
button_explore.grid(row=1, column=0, sticky='n')

# This is where the magic happens
# sv_ttk.set_theme("light")

window.protocol("WM_DELETE_WINDOW", exit)
window.mainloop()