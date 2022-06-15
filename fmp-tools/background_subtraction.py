import argparse

import cv2 as cv


def main():
    
    # Parse args.
    parser = argparse.ArgumentParser(
        description="Subtract background from a colored background")
    parser.add_argument("--png_image",
                        type=str,
                        default=None,
                        help="path to the .png image")
    args = parser.parse_args()
    png_path = args.png_image
    img = cv.imread(png_path)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    h,s,v = cv.split(img)
    cv.imshow('h channel', h)
    cv.imshow('s channel', s)
    cv.imshow('v channel', v)

    # h channel provides best contrast for pink background
    lower_threshold = 40
    upper_threshold = 130
    segmentationn_mask = cv.inRange(img,(lower_threshold, 0, 0), (upper_threshold, 255, 255))

    # invert the image
    segmentationn_mask = cv.bitwise_not(segmentationn_mask)

    # apply blur filter to remove noise
    segmentationn_mask = cv.medianBlur(segmentationn_mask, ksize=5)

    # save image
    file_name = png_path.strip('.png') + '_segmask.png'
    cv.imwrite(file_name, segmentationn_mask)

    # show image
    cv.imshow('segmentation mask', segmentationn_mask)
    cv.waitKey(0)

if __name__ == "__main__":
    main()
