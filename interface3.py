# from tkinter import *
from tkinter import filedialog
import customtkinter as ctk

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
# import sv_ttk
from PIL import Image
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
button_download_segmentation = None
slice_index_label = None
button_compare = None



    
def browseFiles():
    global canvas  
    global slider
    global button_segment
    global segmented_canvas
    global button_overlay
    global button_download_overlay
    global button_download_segmentation
    global slice_index_label
    global button_compare

    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("NIfTI files", "*.nii"),
                                                       ("GZipped NIfTI files", "*.nii.gz"),
                                                       ("PNG files", "*.png"),
                                                       ("all files", "*.*")))
    label_file.configure(text="File Opened: \n" + filename,  font=("Helvetica", 14))
    
    if canvas is not None:
        canvas.get_tk_widget().destroy()

    if segmented_canvas is not None:
        segmented_canvas.get_tk_widget().destroy()
# --------------------------------------------------------------------------------------------
    def segment_slice():
        global segmented_canvas
        global button_overlay
        global button_download_overlay
        global button_download_segmentation
        global button_compare
            

        model = load_h5_model('UNet_wavelet_fusion_150epoch_model_12.h5')
        
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.png':
            img = Image.open(filename)
            original_size = np.array(img).shape
            print('PNG Original size:', original_size)
            # slice_to_segment = read_image(img)
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
        
        # Adjust subplot parameters to remove white border      
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        if segmented_canvas is not None:
            segmented_canvas.get_tk_widget().destroy()

        segmented_canvas = FigureCanvasTkAgg(fig, master=segmented_image_frame)
        segmented_canvas.draw()
        segmented_canvas.get_tk_widget().grid(sticky='nsew')
        number_of_lesions = num_features
        # number_of_lesions_label = Label(actions_frame, text=f"Number of lesions: {number_of_lesions}")
        number_of_lesions_label = ctk.CTkLabel(segmented_image_frame, text=f"Number of lesions: {number_of_lesions}", font=("Helvetica", 20))
        if number_of_lesions_label is not None:
            number_of_lesions_label.pack_forget()
        number_of_lesions_label.pack()
        number_of_lesions_label.place(x=250, y=620,anchor='center')
        def download_segmented():
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            save_filename = f"{base_filename}_slice_{slice_index}_segmented.png"
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     initialfile=save_filename,
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if save_path:
                # Save the segmentation as an image
                Image.fromarray((resized_segmentation * 255).astype(np.uint8)).save(save_path)
                print(f"Segmented image saved to {save_path}")
        
        if button_download_segmentation is None:
            button_download_segmentation = ctk.CTkButton(actions_frame,
                                                    text="Download Segmented Image",
                                                    command=download_segmented)
            button_download_segmentation.grid(row=4, column=0, sticky='n')
            button_download_segmentation.place(x=150, y=400, anchor='center')
        else:
            button_download_segmentation.configure(command=download_segmented)
        # --------------------------------------------------------------------------------------------
        def overlay_images():
            global button_download_overlay
            new_window = ctk.CTkToplevel(window)
            new_window.title("Overlay Image")

            fig, ax = plt.subplots(figsize=[6, 6])
            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(
                # side='TOP', fill='both', 
                expand=1)

            _, ext = os.path.splitext(filename)
            if ext.lower() == '.png':
                img = Image.open(filename)
                original_slice = np.array(img)
            else:
                original_slice = data[:, :, slice_index]
            # original_slice = data[:, :, slice_index]
            
            ax.imshow(original_slice, cmap='gray', origin='lower')
            # Assuming resized_segmentation is your image array
            # Step 1: Create a binary mask
            binary_mask = resized_segmentation == resized_segmentation.max()

            # Step 2: Create a custom RGBA image
            # Initialize an empty RGBA image with the same shape as your mask but with an extra dimension for color
            rgba_image = np.zeros((*binary_mask.shape, 4))

            # Set red color (1, 0, 0) and full opacity (1) for the white pixels
            rgba_image[binary_mask] = [1, 0, 0, 1]

            # Black pixels remain transparent because the default value is [0, 0, 0, 0]

            # Step 3: Display the RGBA image
            ax.imshow(rgba_image, origin='lower')
            # ax.imshow(resized_segmentation, cmap='Reds', alpha=0.3, origin='lower')
            ax.axis("off")
            # Adjust subplot parameters to remove white border
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # Set aspect ratio to equal
            ax.set_aspect('equal')
            canvas.draw()
            
            
            def download_overlay():
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                save_filename = f"{base_filename}_slice_{slice_index}_overlayedSegmentation.png"
                save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                         initialfile=save_filename,
                                                         filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
                if save_path:
                    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    print(f"Overlay image saved to {save_path}")
                    
            if button_download_overlay is None:
                button_download_overlay = ctk.CTkButton(actions_frame,
                                                        text="Download Overlayed Image",
                                                        command=download_overlay)
                button_download_overlay.grid(row=6, column=0, sticky='n')
                button_download_overlay.place(x=150, y=500, anchor='center')
            else:
                button_download_overlay.configure(command=download_overlay)

        # --------------------------------------------------------------------------------------------
        if button_overlay is None:
            button_overlay = ctk.CTkButton(actions_frame, 
                                text="Overlay Images",
                                command=overlay_images) 
            button_overlay.grid(row=5, column=0, sticky='n')
            button_overlay.place(x=150, y=450, anchor='center')
        else:
            button_overlay.configure(command=overlay_images)
            
        def compare_with_old():
            # load an image 
            filename_old = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("PNG files", "*.png"),
                                                       ("all files", "*.*")))
            
            # display the semnetation of the old image and the new image on top of each other
            new_window = ctk.CTkToplevel(window)
            new_window.title("comapre Images")
            fig, ax = plt.subplots(figsize=[6, 6])
            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.draw()
            canvas.get_tk_widget().pack(
                # side='TOP', fill='both', 
                expand=1)
            
            _, ext = os.path.splitext(filename_old)
            
            old_slice_img = Image.open(filename_old)
            # Resize the image to match 'resized_segmentation' dimensions
            resized_old_slice_img = old_slice_img.resize((resized_segmentation.shape[1], resized_segmentation.shape[0]))

            # Optionally, convert the resized image to a NumPy array if you need to process it further
            resized_old_slice = np.array(resized_old_slice_img)
            ax.imshow(resized_old_slice, cmap='gray', origin='lower')
            
            # Step 1: Create a binary mask
            binary_mask = resized_segmentation == resized_segmentation.max()

            # Step 2: Create a custom RGBA image
            # Initialize an empty RGBA image with the same shape as your mask but with an extra dimension for color
            rgba_image = np.zeros((*binary_mask.shape, 4))

            # Set red color (1, 0, 0) and full opacity (1) for the white pixels
            rgba_image[binary_mask] = [0, 1, 0, 0.5]

            # Black pixels remain transparent because the default value is [0, 0, 0, 0]

            # Step 3: Display the RGBA image
            ax.imshow(rgba_image, origin='lower')
            # ax.imshow(resized_segmentation, cmap='Reds', alpha=0.3, origin='lower')
            ax.axis("off")
            # Adjust subplot parameters to remove white border
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # Set aspect ratio to equal
            ax.set_aspect('equal')
            canvas.draw()
            
        if button_compare is None:
            button_compare = ctk.CTkButton(actions_frame, 
                                text="Compare with old",
                                command=compare_with_old) 
            button_compare.grid(row=7, column=0, sticky='n')
            button_compare.place(x=150, y=650, anchor='center')
        else:
            button_compare.configure(command=compare_with_old)
            
            
            
            
# Load the image
    _, ext = os.path.splitext(filename)
    if ext.lower() == '.png':
        img = Image.open(filename)
        data = np.array(img)
    else:
        img = nib.load(filename)
        data = img.get_fdata()


    # img = nib.load(filename)
    # data = img.get_fdata()
    
    print('Shape:', data.shape)
    print('Max:', data.max())
    print('Min:', data.min())
    
    # Remove the old slider
    if slider is not None:
        slider.pack_forget()
        
    if slice_index_label is not None:
        slice_index_label.pack_forget()

    # Create a slider
    # slider = Scale(actions_frame, from_=0, to=data.shape[2] - 1, orient=HORIZONTAL, command=lambda s: update_slice(s, data, im))
    # slider.grid(row=2, column=0, sticky='n')
    # Create a label to display the current slice index
    

    # Modify the slider command to update the label text
    # slider = ctk.CTkSlider(actions_frame, from_=0, to=data.shape[2] - 1, 
    #                        orient='horizontal', 
    #                command=lambda s: (update_slice(s, data, im), slice_index_label.configure(text=f"Slice index: {s}")))
    _, ext = os.path.splitext(filename)
    if ext.lower() != '.png':
        slice_index_label = ctk.CTkLabel(actions_frame, text="Slice index: 0", font=("Helvetica", 16))
        slice_index_label.grid(row=2, column=0, sticky='n')
        slice_index_label.place(x=150, y=300, anchor='center')
        slider = ctk.CTkSlider(actions_frame, from_=0,
                            to=data.shape[2] - 1, 
                            number_of_steps=data.shape[2] - 1,
                            command=lambda s: (update_slice(s, data, im), slice_index_label.configure(text=f"Slice index: {s}")))
        slider.set(0)
        slider.grid(row=3, column=0, sticky='n')
        slider.place(x=150, y=350, anchor='center')
    
    # Remove the old segment button
    if button_segment is not None:
        button_segment.pack_forget()
    
    button_segment = ctk.CTkButton(actions_frame, 
                        text="segment this image",
                        command=segment_slice) 
    button_segment.grid(row=4, column=0, sticky='n')
    button_segment.place(x=150, y=250, anchor='center')
    
    # Create a canvas and add it to the window
    # display the canvas on the left side of the window
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
    
    # Adjust subplot parameters to remove white border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Set aspect ratio to equal
    ax.set_aspect('equal')


    canvas = FigureCanvasTkAgg(fig, master=original_image_frame)
    canvas.get_tk_widget().grid(sticky='nsew')
    canvas.draw()

    
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
# Create the window an layout
# ======================================================================================================================
window = ctk.CTk()

window.title("MS leisons Segmentation")
window.after(0, lambda:window.state('zoomed')) 
# window.state('zoomed')

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


window.rowconfigure(0, weight=3)
window.rowconfigure(1, weight=20)
window.columnconfigure(0, weight=3)
window.columnconfigure(1, weight=9)
window.columnconfigure(2, weight=9)






main_label_frame = ctk.CTkFrame(window, border_width=1)
main_label_frame.grid(row=0, column=0, columnspan=3, sticky='nsew')
main_label = ctk.CTkLabel(main_label_frame, text="SEP Detect Interface",  font=("Helvetica", 28,"bold"))
main_label.pack(fill='both', expand=True)

light_image = Image.open("CDTA Logo white.png")
dark_image = Image.open("CDTA Logo white.png")
cdta_logo = ctk.CTkImage(light_image=light_image, 
                         dark_image=dark_image, 
                         size=(150, 30))
cdta_logo = ctk.CTkLabel(main_label_frame, image=cdta_logo, text="")
cdta_logo.pack(side='left')
cdta_logo.place(x=10, y=30, anchor='nw')

light_image = Image.open("AC2_Logo_NEW white.png")
dark_image = Image.open("AC2_Logo_NEW white.png")
cdta_logo = ctk.CTkImage(light_image=light_image, 
                         dark_image=dark_image, 
                         size=(65, 95))
cdta_logo = ctk.CTkLabel(main_label_frame, image=cdta_logo, text="")
cdta_logo.pack(side='left')
cdta_logo.place(x=170, y=5, anchor='nw')

light_image = Image.open("esi-logo-white.png")
dark_image = Image.open("esi-logo-white.png")
cdta_logo = ctk.CTkImage(light_image=light_image, 
                         dark_image=dark_image, 
                         size=(100, 100))
cdta_logo = ctk.CTkLabel(main_label_frame, image=cdta_logo, text="")
cdta_logo.pack(side='right')
cdta_logo.place(x=1420, y=5, anchor='nw')

actions_frame = ctk.CTkFrame(window, border_width=1)
actions_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
actions_frame.grid_propagate(False)


original_image_frame = ctk.CTkFrame(window, border_width=1)
original_image_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)
original_image_frame.grid_propagate(False)

segmented_image_frame = ctk.CTkFrame(window, border_width=1)
segmented_image_frame.grid(row=1, column=2, sticky='nsew', padx=10, pady=10)
segmented_image_frame.grid_propagate(False)


actions_frame.rowconfigure(0, weight=1)
actions_frame.rowconfigure(1, weight=1)
actions_frame.rowconfigure(2, weight=1)
actions_frame.rowconfigure(3, weight=1)
actions_frame.rowconfigure(4, weight=1)
actions_frame.rowconfigure(5, weight=1)
actions_frame.rowconfigure(6, weight=1)
actions_frame.rowconfigure(7, weight=10)


actions_frame.grid_propagate(False)

label_file = ctk.CTkLabel(actions_frame,
                    text = "Select a file",
                    wraplength=200,
                    font=("Helvetica", 14),
                    justify='center'
                    )
label_file.grid(row=0, column=0, sticky='ew', pady=10)
label_file.place(x=150, y=90, anchor='center')

button_explore = ctk.CTkButton(actions_frame, 
                        text="Browse Files",
                        command=browseFiles
                        ) 
button_explore.grid(row=1, column=0, sticky='n')
button_explore.place(x=150, y=150, anchor='center')

# This is where the magic happens
# sv_ttk.set_theme("light")

window.protocol("WM_DELETE_WINDOW", exit)
window.mainloop()