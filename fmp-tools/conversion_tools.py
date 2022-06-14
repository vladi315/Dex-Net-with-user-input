
import numpy as np
from autolab_core import CameraIntrinsics, DepthImage
from PIL import Image


def convert_png_to_npy(png_path):
    # sample.png is the name of the image
    # file and assuming that it is uploaded
    # in the current directory or we need
    # to give the path
    image = Image.open(png_path)
    
    # summarize some details about the image
    print(image.format)
    print(image.size)
    print(image.mode)

    
    # np.asarray() class is used to convert
    # PIL images into NumPy arrays
    npy_array = np.asarray(image)
    
    # <class 'numpy.ndarray'>
    # print(type(npy_array))
    
    #  shape
    # print(npy_array.shape)

    return npy_array


def convert_depth_to_dexnet_format(npy_array):
    """
    The depth images need to be converted to meters with float32 dtype. 
    Technically it will run as you have found out, but the object looks extremely far away to GQ-CNN 
    when you give it the raw uint16 images. To the best of my knowledge, the depth image in uint16 
    format contains the depth in millimeters, so you can convert to the correct size by dividing by 
    1000.0.(from https://github.com/BerkeleyAutomation/gqcnn/issues/13)

    Convert to 640 x 480  px if necessary beforehand manually!
    """
    #TODO: RENAME TO "DEPTH"
    npy_array = np.float32(npy_array)
    npy_array = np.expand_dims(npy_array, axis=2) 
    npy_array = npy_array/1000

    # camera_intr_filename = "data/calib/realsense/realsense.intr"
    # camera_intr = CameraIntrinsics.load(camera_intr_filename)
    # depth_im = DepthImage(npy_array, frame=camera_intr.frame)
    return npy_array


def show_npy_image(npy_array):
    return 0


def save_as_npy(npy_array, npy_path):
    #save 
    np.save(npy_path, npy_array)


def main():
    png_path = '/home/vladislav/gqcnn/fmp-tools/0003_depth_raw.png'
    npy_path = png_path[:-8] + '.npy'

    # convert png to npy format of dexnet
    npy_array = convert_png_to_npy(png_path)
    npy_array = convert_depth_to_dexnet_format(npy_array)
    # show npy image 
    show_npy_image(npy_array)
    # save
    save_as_npy(npy_array, npy_path)

    return 0

if __name__ == "__main__":
    main()
