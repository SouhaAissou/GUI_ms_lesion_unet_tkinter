from tkinter import *
from tkinter import filedialog
import customtkinter as ctk

import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import os
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import io
import cv2

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
button_download_overlay = None
button_load_segmented = None
slice_index_label = None
filename_old = None
filename_new = None
segmentation_old = None
segmentation_new = None
loaded_segmented_img = None

def browseFilesOld():
    global filename_old
    filename_old = filedialog.askopenfilename(initialdir = "/", 
                                              title = "Select Old Scan File", 
                                              filetypes = (("NIfTI files", "*.nii"),
                                                           ("GZipped NIfTI files", "*.nii.gz"),
                                                           ("PNG files", "*.png"),
                                                           ("all files", "*.*")))
    label_file_old.configure(text="Old Scan File: \n" + filename_old, font=("Helvetica", 14))
    display_image(filename_old, old=True)

def browseFilesNew():
    global filename_new
    filename_new = filedialog.askopenfilename(initialdir = "/", 
                                              title = "Select New Scan File", 
                                              filetypes = (("NIfTI files", "*.nii"),
                                                           ("GZipped NIfTI files", "*.nii.gz"),
                                                           ("PNG files", "*.png"),
                                                           ("all files", "*.*")))
    label_file_new.configure(text="New Scan File: \n" + filename_new, font=("Helvetica", 14))
    display_image(filename_new, old=False)

def load_segmented():
    global loaded_segmented_img
    file_path = filedialog.askopenfilename(initialdir = "/", 
                                           title = "Select Segmented Image File", 
                                           filetypes = (("PNG files", "*.png"),
                                                        ("All files", "*.*")))
    if file_path:
        loaded_segmented_img = Image.open(file_path).convert('L')
        loaded_segmented_img = np.array(loaded_segmented_img)
        display_loaded_segmented_image()

def display_loaded_segmented_image():
    global loaded_segmented_img
    if loaded_segmented_img is not None:
        fig, ax = plt.subplots(figsize=[4, 4])
        ax.imshow(loaded_segmented_img, cmap="gray", origin="lower")
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_aspect('equal')

        if segmented_canvas is not None:
            segmented_canvas.get_tk_widget().destroy()
        
        segmented_canvas = FigureCanvasTkAgg(fig, master=segmented_image_frame)
        segmented_canvas.draw()
        segmented_canvas.get_tk_widget().grid(sticky='nsew')

def display_image(filename, old=True):
    global canvas
    global slider
    global button_segment
    global segmented_canvas
    global button_overlay
    global button_download_overlay
    global button_load_segmented
    global slice_index_label
    global slice_index

    if canvas is not None:
        canvas.get_tk_widget().destroy()

    if segmented_canvas is not None:
        segmented_canvas.get_tk_widget().destroy()

    def segment_slice():
        global segmented_canvas
        global button_overlay
        global button_download_overlay
        global segmentation_old
        global segmentation_new

        model = load_h5_model('UNet_wavelet_fusion_150epoch_model_12.h5')
        
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.png':
            img = Image.open(filename)
            original_size = np.array(img).shape
            img = img.convert('L').resize((128, 128))
            img_array = np.array(img)
            img_array = img_array.astype(np.float32)
            img_array = img_array / 255.0
            img_array = np.clip(img_array, 0, 1)
            slice_to_segment = img_array
            slice_to_segment = np.expand_dims(slice_to_segment, axis=-1)

            segmentation = model.predict(slice_to_segment[None, ..., None])
            resized_segmentation =  zoom(segmentation[0, :, :, 0], (original_size[0] / segmentation.shape[1], original_size[1] / segmentation.shape[2]))
            
        else:
            slice_index = int(slider.get())
            original_size = data[:, :, slice_index].shape
            slice_to_segment = read_image(nifti_slice_to_png(filename, slice_index))
            slice_to_segment = np.expand_dims(slice_to_segment, axis=-1)

            segmentation = model.predict(slice_to_segment[None, ..., None])
            resized_segmentation = zoom(segmentation[0, :, :, 0], (original_size[0] / segmentation.shape[1], original_size[1] / segmentation.shape[2]))

        threshold = 0.5  
        resized_segmentation = (resized_segmentation > threshold).astype(int)

        if old:
            segmentation_old = resized_segmentation
        else:
            segmentation_new = resized_segmentation

        frame_width = segmented_image_frame.winfo_width()
        frame_height = segmented_image_frame.winfo_height()

        dpi = min(frame_width / 4, frame_height / 4)
        
        fig, ax = plt.subplots( figsize=[4, 4], dpi=dpi)
        ax.imshow(resized_segmentation, cmap="gray", origin="lower")
        ax.axis('off')
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_aspect('equal')
        
        if segmented_canvas is not None:
            segmented_canvas.get_tk_widget().destroy()

        segmented_canvas = FigureCanvasTkAgg(fig, master=segmented_image_frame)
        segmented_canvas.draw()
        segmented_canvas.get_tk_widget().grid(sticky='nsew')

        # Download segmented image function
        def download_segmented():
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            save_filename = f"{base_filename}_slice_{slice_index}_segmented.png"
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     initialfile=save_filename,
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if save_path:
                Image.fromarray((resized_segmentation * 255).astype(np.uint8)).save(save_path)
                print(f"Segmented image saved to {save_path}")

        if button_download_overlay is None:
            button_download_overlay = ctk.CTkButton(actions_frame,
                                                    text="Download Segmented Image",
                                                    command=download_segmented)
            button_download_overlay.grid(row=7, column=0, sticky='n')
            button_download_overlay.place(x=150, y=600, anchor='center')
        else:
            button_download_overlay.configure(command=download_segmented)

        # Overlay images function
        def overlay_images():
            new_window = ctk.CTkToplevel(window)
            new_window.title("Overlay Image")

            fig, ax = plt.subplots(figsize=[6, 6])
            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=1)

            _, ext = os.path.splitext(filename)
            if ext.lower() == '.png':
                img = Image.open(filename)
                original_slice = np.array(img)
            else:
                original_slice = data[:, :, slice_index]
                
            ax.imshow(original_slice, cmap='gray', origin='lower')
            
            if old:
                binary_mask_old = segmentation_old == segmentation_old.max()
                rgba_image_old = np.zeros((*binary_mask_old.shape, 4), dtype=np.uint8)
                rgba_image_old[binary_mask_old] = [255, 0, 0, 255]
                ax.imshow(rgba_image_old, cmap="Reds", origin="lower", alpha=0.3)
            else:
                binary_mask_new = segmentation_new == segmentation_new.max()
                rgba_image_new = np.zeros((*binary_mask_new.shape, 4), dtype=np.uint8)
                rgba_image_new[binary_mask_new] = [0, 0, 255, 255]
                ax.imshow(rgba_image_new, cmap="Blues", origin="lower", alpha=0.3)

            canvas.draw()

        if button_overlay is None:
            button_overlay = ctk.CTkButton(actions_frame, text="Overlay Images", command=overlay_images)
            button_overlay.grid(row=6, column=0, sticky='n')
            button_overlay.place(x=50, y=600, anchor='center')
        else:
            button_overlay.configure(command=overlay_images)

    _, ext = os.path.splitext(filename)
    if ext.lower() == '.nii' or ext.lower() == '.nii.gz':
        img = nib.load(filename)
        data = img.get_fdata()
        middle_slice = data.shape[2] // 2
        slice_index = middle_slice

        fig, ax = plt.subplots(figsize=[4, 4])
        ax.imshow(data[:, :, slice_index], cmap="gray", origin="lower")
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_aspect('equal')

        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky='nsew')

        if slider is not None:
            slider.destroy()
        if slice_index_label is not None:
            slice_index_label.destroy()

        slice_index_label = ctk.CTkLabel(image_frame, text=f"Slice Index: {slice_index}", font=("Helvetica", 20))
        slice_index_label.pack()
        slice_index_label.place(x=250, y=620, anchor='center')

        slider = ctk.CTkSlider(master=actions_frame, from_=0, to=data.shape[2] - 1, command=update_image)
        slider.set(middle_slice)
        slider.grid(row=4, column=0, sticky='ew')
        slider_label = ctk.CTkLabel(actions_frame, text='Slice Index')
        slider_label.grid(row=3, column=0, sticky='ew')
        
        if button_segment is None:
            button_segment = ctk.CTkButton(actions_frame, text="Segment Slice", command=segment_slice) 
            button_segment.grid(row=5, column=0, sticky='n')
        else:
            button_segment.configure(command=segment_slice)

    elif ext.lower() == '.png':
        img = Image.open(filename)
        fig, ax = plt.subplots(figsize=[4, 4])
        ax.imshow(img, cmap="gray", origin="lower")
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_aspect('equal')

        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(sticky='nsew')
        
        if slider is not None:
            slider.destroy()
        if slice_index_label is not None:
            slice_index_label.destroy()

        if button_segment is None:
            button_segment = ctk.CTkButton(actions_frame, text="Segment Image", command=segment_slice) 
            button_segment.grid(row=5, column=0, sticky='n')
        else:
            button_segment.configure(command=segment_slice)

def update_image(val):
    slice_index = int(float(val))
    slice_index_label.configure(text=f"Slice Index: {slice_index}")

    if canvas is not None:
        canvas.get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=[4, 4])
    ax.imshow(data[:, :, slice_index], cmap="gray", origin="lower")
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_aspect('equal')

    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(sticky='nsew')

# ======================================================================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Lesion Segmentation GUI")

frame_1 = ctk.CTkFrame(window, corner_radius=10)
frame_1.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

label_file_old = ctk.CTkLabel(frame_1, text="No old file selected", width=70, height=120, 
                          fg_color=("white", "gray38"), justify="left", anchor="center", font=("Helvetica", 14))
label_file_old.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

button_explore_old = ctk.CTkButton(frame_1, text="Browse Old Scan Files", command=browseFilesOld)
button_explore_old.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

label_file_new = ctk.CTkLabel(frame_1, text="No new file selected", width=70, height=120, 
                          fg_color=("white", "gray38"), justify="left", anchor="center", font=("Helvetica", 14))
label_file_new.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

button_explore_new = ctk.CTkButton(frame_1, text="Browse New Scan Files", command=browseFilesNew)
button_explore_new.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

button_load_segmented = ctk.CTkButton(frame_1, text="Load Old Segmented Image", command=load_segmented)
button_load_segmented.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

image_frame = ctk.CTkFrame(window, corner_radius=10)
image_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

segmented_image_frame = ctk.CTkFrame(window, corner_radius=10)
segmented_image_frame.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")

actions_frame = ctk.CTkFrame(window, corner_radius=10)
actions_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=20, sticky="nsew")

window.mainloop()
