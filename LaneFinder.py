import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5,draw_line_segments = False):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    image_height = img.shape[0]

    if draw_line_segments:
        draw_lineSegments(lines, img)
    else:
        left_line, right_line = average_lane_lines(lines,0.6*image_height,image_height)

        if not (left_line is None):
            l_x1x1, l_x2y2 = left_line
            cv2.line(img, l_x1x1, l_x2y2, color, thickness)
        if not (right_line is None):
            r_x1x1, r_x2y2 = right_line
            cv2.line(img, r_x1x1, r_x2y2, color, thickness)



def average_lane_lines(lines,y_low,y_high):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    if lines is None:
        return None,None

    for line in lines:
        if line is None:
            continue
        for x1, y1, x2, y2 in line:

            if x2 == x1 or y2 == y1:  # vertical or horizontal lines
                continue

            slope = np.float32((y2 - y1)) / np.float32((x2 - x1))
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane_slope_intercept = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane_slope_intercept = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    left_lane_points = get_line_points(left_lane_slope_intercept,y_low,y_high)
    right_lane_points = get_line_points(right_lane_slope_intercept, y_low, y_high)
    return left_lane_points, right_lane_points

def draw_lineSegments(lines,img):
    if lines is None:
        return 
    for line in lines:
        if line is None:
            continue

        for x1, y1, x2, y2 in line:
            if x2 == x1 or y2 == y1:  # vertical or horizontal lines
                continue

            cv2.line(img, (x1,y1), (x2,y2), [255, 0, 0], 2)


def get_line_points(line,sample_y1, sample_y2):
    """ Convert a line represented by slope and intercept into points """
    if line is None:
        return None

    slope, intercept = line

    # convert to int to be used as pixel points
    x1 = int((sample_y1 - intercept) / slope)
    x2 = int((sample_y2 - intercept) / slope)
    y1 = int(sample_y1)
    y2 = int(sample_y2)

    return ((x1, y1), (x2, y2))

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def process_image(image):
    """ detect lane lines on the image and highlight it with red lines"""
    # Grab the x and y size of the image
    ysize = image.shape[0]
    xsize = image.shape[1]

    color_mask = color_select(image, 190, 190, 10)

    gray = grayscale(color_mask)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)
    vertices = np.array([[[xsize / 2 - 50, 0.65 * ysize], [xsize / 2 + 50, 0.65 * ysize], [xsize, ysize], [0, ysize]]],
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    line_image = hough_lines(masked_edges, 1, np.pi / 180, 20, 15, 4)
    color_image = image.copy()
    # Draw the lines on the color image
    return cv2.addWeighted(color_image, 0.8, line_image, 1, 0)

def color_select(image,red_threshold,green_threshold,blue_threshold):
    filtered_image = np.copy(image)
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                 | (image[:, :, 1] < rgb_threshold[1]) \
                 | (image[:, :, 2] < rgb_threshold[2])
    filtered_image[thresholds] = [0, 0, 0]

    return filtered_image

def Process_Images():
    image_list = os.listdir("test_images/")
    for entry in image_list:
        image = mpimg.imread('test_images/' + entry)

        output_image = process_image(image)
        mpimg.imsave("test_images_output/" + entry, output_image)

def Process_Videos():
    videos_list = os.listdir("test_videos/")

    for entry in videos_list:
        vclip = VideoFileClip('test_videos/' + entry)

        processed_clip = vclip.fl_image(process_image)

        processed_clip.write_videofile('test_videos_output/'+entry, audio=False)


Process_Videos()
