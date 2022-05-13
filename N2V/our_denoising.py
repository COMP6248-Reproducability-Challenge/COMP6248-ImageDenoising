import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# We import all our dependencies.
from n2v.models import N2V

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

rootpath = "/home/ben/Documents/University Of Southampton/COMP6248 Deep Learning/Coursework/Datasets/Noisy Datasets (.npy)/"
rgb_datasets = ["CBSD68", "Kodak"]
greyscale_datasets = ["BSD68", "Set12"]
noise_types = ["gamma_50", "gamma_100", "gaussian_25", "gaussian_50", "poisson_0.01", "poisson_0.05"]

def denoise_images_from_folder(model, path: str, savepath):
    images = [(filename, np.load(path + filename)) for filename in os.listdir(path)]
    preds = [(filename, model.predict(image, axes="XYC")) for filename, image in images]
    #print(preds)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print(f"Made new directory {savepath}")
    for filename, pred in preds:
        print(f"Writing to {savepath}/{filename}.jpg")
        plt.imshow(pred)
        plt.show()
        cv2.imwrite(f"{savepath}/{filename}.jpg", pred)

def denoise_bw_images_from_folder(model, path: str, savepath):
    images = [(filename, np.load(path + filename)) for filename in os.listdir(path)]
    preds = [(filename, model.predict(image.astype(np.float32), 'YX', tta=False)) for filename, image in images]
    #print(preds)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print(f"Made new directory {savepath}")
    for filename, pred in preds:
        print(f"Writing to {savepath}/{filename}.jpg")
        cv2.imwrite(f"{savepath}/{filename}.jpg", pred)


# A previously trained model is loaded by creating a new N2V-object without providing a 'config'.
model_name = 'n2v_2D'
basedir = 'examples/2D/denoising2D_RGB/models'
rgb_model = N2V(config=None, name=model_name, basedir=basedir)

model_name = 'BSD68_reproducability_5x5'
basedir = 'examples/2D/denoising2D_BSD68/models'
bw_model = N2V(config=None, name=model_name, basedir=basedir)


for dataset in rgb_datasets:
    for noise_type in noise_types:
        source_path = f"{rootpath}{dataset}/{str.lower(dataset)}_{noise_type}/"
        output_path = f"{rootpath}Denoised/Noise2Void/{str.lower(dataset)}_{noise_type}"
        denoise_images_from_folder(rgb_model, source_path, output_path)

for dataset in greyscale_datasets:
    for noise_type in noise_types:
        source_path = f"{rootpath}{dataset}/{str.lower(dataset)}_{noise_type}/"
        output_path = f"{rootpath}Denoised/Noise2Void/{str.lower(dataset)}_{noise_type}"
        denoise_bw_images_from_folder(bw_model, source_path, output_path)