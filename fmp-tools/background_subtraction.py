from plantcv import plantcv as pcv

# Set global debug behavior to None (default), "print" (to file), 
# or "plot" (Jupyter Notebooks or X11)

# pcv.params.debug = "plot"

# plant_img = "/home/vladislav/Downloads/0000_image_obj.png"
# b_img = "/home/vladislav/Downloads/0000_image.png"

def main():
    
    # Parse args.
    parser = argparse.ArgumentParser(
        description="Subtract background from a (pink) colored background")
    parser.add_argument("--png_image",
                        type=str,
                        default=None,
                        help="path to the .png image")
    args = parser.parse_args()
    png_path = args.png_image
    img, _, _ = pcv.readimage(filename=png_path) 

    # select "print" to save images, "plot to display them"
    pcv.params.debug = "print" 

    # apply hsv filters
    h = pcv.rgb2gray_hsv(rgb_img=img, channel='h')
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    v = pcv.rgb2gray_hsv(rgb_img=img, channel='v')

    # select filter that provides best contrast
    h_thresh, _ = pcv.threshold.custom_range(img=h, lower_thresh=[40], upper_thresh=[130], channel='gray')
    h_mblur = pcv.median_blur(gray_img=h_thresh, ksize=5)

if __name__ == "__main__":
    main()
