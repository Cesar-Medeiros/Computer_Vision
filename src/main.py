import argparse
import image_acquisition as img_acq
import image_manipulation as img_man
import detection as d
import user_interface as usr_int
import cv2 as cv
import numpy as np
import util

IMAGE_SCALE_WIDTH = 500
IMAGE_SCALE_HEIGHT = 500

def main(img, useHough):

    # Scale image maintaining w/h ratio
    img = img_man.scale_image(img, IMAGE_SCALE_WIDTH, IMAGE_SCALE_HEIGHT)

    # Show image
    cv.imshow("Img", img)
    
    img_eq = img

    # Smooth image maintaing edges
    bgr = cv.cvtColor(img_eq, cv.COLOR_BGRA2BGR)
    img_eq = cv.bilateralFilter(bgr, 10, 30, 30)

    # Remove small noise from edges left from bilateral filter
    img_eq = cv.medianBlur(img_eq, 5)

    # Equalize saturation
    img_eq = img_man.hist_eq_sat(img_eq)
    cv.imshow("Img_eq", img_eq)


    blue_range = d.ColorRange([104, 200, 75], [126, 255, 255])
    blue_detection = d.ShapeDetection(util.Color.BLUE, [blue_range])

    red_range1 = d.ColorRange([0, 200, 75], [5, 255, 255])
    red_range2 = d.ColorRange([170, 200, 75], [180, 255, 255])
    red_detection = d.ShapeDetection(util.Color.RED, [red_range1, red_range2])

    color_detectors = [blue_detection, red_detection]

    masks = []

    for color_detector in color_detectors:
        mask = color_detector.segment_color(img_eq)
        masks.append(mask)
        cv.imshow(util.getColorName(color_detector.color) + "Mask", mask)


    # Edge canvas
    height, width, _ = img.shape
    edge_canvas = np.zeros((height, width, 1), np.uint8)

    # Find circles with Hough Transform, if chosen
    if (useHough):
        hough = d.HoughCircles(img_eq)
        circles = hough.getCircles()

        for circle in circles:
            
            section = img_man.circular_section(img_eq, circle)

            non_zeros = []

            for color_detector in color_detectors:
                mask = color_detector.segment_color(section)
                non_zeros.append(cv.countNonZero(mask))
            
            maximum, index = util.getMax(non_zeros)

            if(maximum == 0 or index == -1):
                continue

            x,y,r = circle
            label = util.getColorName(color_detectors[index].color) + ' Circle'

            cv.circle(img, (x, y), r, (255, 0, 0), 3)
            cv.circle(img, (x, y), 2, (255, 0, 255), 3)
            cv.putText(img, label, (x - 40, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)



    for i, color_detector in enumerate(color_detectors):
        shapes = color_detector.getShape(masks[i], useHough)
        color_detector.outputEdges(shapes, edge_canvas)
        color_detector.outputResult(shapes, img)
    

    cv.imshow("Edges", edge_canvas)
    cv.imshow("Result", img)

    cv.waitKey(0)
    cv.destroyAllWindows()



    
if __name__ == '__main__':
    print(__doc__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--f', help="The image filepath")

    img = None
    file = parser.parse_args().f

    if (file):
        img = img_acq.load_image(file)
    else:
        img = img_acq.capture_frame()


    # Ask if Circle Hough Transform is to be chosen
    useHough = usr_int.optionMenu()

    main(img, useHough)