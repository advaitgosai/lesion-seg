import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

smooth=0

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jacard_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    union = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1) - intersection
    jacard = (intersection + smooth) / (union + smooth)
    return jacard

def predict():
    model_dir = "models/"
    im_width = 256
    im_height = 256

    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.hdf5')]

    for model_file in model_files:
        # Load the saved model
        model = tf.keras.models.load_model(model_file, custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef, 'jacard_coef': jacard_coef})
        
        # Define the path to the parent directory containing all the validation image folders
        parent_dir = "preprocessed/2D/"

        # Define the output directory to save the predictions
        model_name = model_file.split("/")[1].split(".")[0]
        output_dir = f'predicted/2D/{model_name}'

        # create output folder if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Loop through all validation image folders and make predictions
        for folder_name in os.listdir(parent_dir):
            # Define the path to the validation image folder
            val_image_path = os.path.join(parent_dir, folder_name)

            # Load the validation images and sort them
            val_images = []
            for i in os.listdir(val_image_path):
                if i.endswith('.jpg'): # Modify to fit your file format
                    val_images.append(os.path.join(val_image_path, i))
            val_images.sort()

            # Create a dataframe to store the validation image paths
            df_val = pd.DataFrame()
            df_val['val_img'] = val_images

            # Create a folder in the output directory with the same name as the validation image folder
            output_folder = os.path.join(output_dir, folder_name)
            os.makedirs(output_folder, exist_ok=True)

            # Loop through all validation images and make predictions
            for i in range(len(df_val)):
                # Load the validation image
                val_img = cv2.imread(df_val['val_img'].iloc[i])
                # Resize and normalize the image
                val_img = cv2.resize(val_img, (im_height, im_width))
                val_img = val_img / 255
                val_img = val_img[np.newaxis, :, :, :]

                # Make a prediction using the loaded model
                pred = model.predict(val_img)

                # Save the predicted image to the output folder with the same name as the original file
                file_name = os.path.basename(df_val['val_img'].iloc[i])
                output_path = os.path.join(output_folder, file_name)
                plt.imsave(output_path, np.squeeze(pred) > .5, cmap='gray')



