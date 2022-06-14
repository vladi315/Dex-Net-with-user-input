from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file), 
# or "plot" (Jupyter Notebooks or X11)

# pcv.params.debug = "plot"

# plant_img = "/home/vladislav/Downloads/0000_image_obj.png"
# b_img = "/home/vladislav/Downloads/0000_image.png"


pcv.params.debug = "print"

img, path, filename = pcv.readimage(filename="/home/vladislav/gqcnn/fmp-tools/0004_image.png") #0004_image.png

# img, path, filename = pcv.readimage(filename="/home/vladislav/Downloads/alllight/alllighthingPollock/0000_image.png")
# img_bkgrd, bkgrd_path, bkgrd_filename = pcv.readimage(filename="/home/vladislav/Downloads/alllight/alllighthingPollockOnlyBackground/0000_image.png")

# Create a foreground mask from both images 
# fgmask = pcv.background_subtraction(foreground_image=img, background_image=img_bkgrd)

# 
h = pcv.rgb2gray_hsv(rgb_img=img, channel='h')
s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
v = pcv.rgb2gray_hsv(rgb_img=img, channel='v')

# best_contrast = h

# h_thresh = pcv.threshold.binary(gray_img=h, threshold=130, max_value=255, object_type='dark')
h_thresh, _ = pcv.threshold.custom_range(img=h, lower_thresh=[40], upper_thresh=[130], channel='gray')

h_mblur = pcv.median_blur(gray_img=h_thresh, ksize=5)

# s_thresh = pcv.threshold.binary(gray_img=s, threshold=50, max_value=255, object_type='light')
# s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)

# s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, max_value=255, object_type='light')
# s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
# gaussian_img = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)


# b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
# b_thresh = pcv.threshold.binary(gray_img=b, threshold=160, max_value=255, object_type='light')
# bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)




test=1
