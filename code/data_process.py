import random
import glob
import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
# from google.colab.patches import cv2_imshow
from shapely.geometry import Point
import tensorflow as tf
from tensorflow.keras.applications import resnet50, mobilenet_v2, vgg16


class DataProcessor():
    def __init__(self, args, plot_light=False, plot_extensive=False, show_prints=False):

        self.min_height = args['height']
        self.min_width = args['width']
        self.rotate = args['rotate']
        self.plot_light = plot_light
        self.plot_extensive = plot_extensive
        self.show_prints = show_prints
        self.img_name = None


    def preprocess_pipeline(self, image_path, bbox):
        '''
        Defines the preprocess pipeline for a SINGLE IMAGE.
        Inputs: 
            - image_path: path where the image is stored
            - bbox: array with location and dimensions of the bounding box in the image
        Outputs:
            - output_image: image after the pre-processing pipeline of {rotation},
            padding and cropping to standard size
            - bbox_sized: bounding box array tranformed during the pre-processing chain
        '''
        image_path = image_path.numpy().decode('utf-8')
        bbox = bbox.numpy()
        if self.show_prints:
            print("[START]: preprocess_pipeline")
            print(image_path, bbox)
        self.img_name = os.path.basename(image_path)[:-4]

        # loads images as BGR in float32
        im = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB

        if self.plot_extensive:
            print("Before:", image.shape)
            plt.imshow(image)

        if self.rotate:
            # perform rotation to the images
            image, bbox_rotated = self.rotation(image_path, bbox)
        else:
            # look for bounding box
            bbox_rotated = None if (bbox == 0).all() else bbox

        # perform padding and then cropping to the images
        padded_image, bbox_padded = self.padding(image, bbox_rotated)
        sized_image, bbox_sized = self.random_crop(padded_image, bbox_padded)

        if self.show_prints:
            print("[END]: preprocess_pipeline")
        return sized_image, bbox_sized


    def padding(self, image, bbox):
        '''Takes an input image and returns a padded version of it with
        the required dimensions to reach self.min_height, self.min_width'''
        h, w = image.shape[:2]
        if h >= self.min_height and w >= self.min_width:
            return image, bbox

        new_values = None
        h1 = (self.min_height - h)//2 if (self.min_height - h)//2 > 0 else 0
        w1 = (self.min_width - w)//2 if (self.min_width - w)//2 > 0 else 0
        padded_img = cv2.copyMakeBorder(image,
                                        top = h1+1 if (self.min_height-(h+2*h1)) == 1 else h1,
                                        bottom = h1,
                                        left = w1+1 if (self.min_width-(w+2*w1)) == 1 else w1,
                                        right = w1,
                                        borderType = cv2.BORDER_CONSTANT,
                                        value = [0.0, 0.0, 0.0])
        if bbox is not None:
            x_left = (int)(bbox[0])
            y_top = (int)(bbox[1])
            bb_width = (int)(bbox[2])
            bb_height = (int)(bbox[3])

            # translate the x_left and y_top values according to the pad
            x_left += w1+1 if (self.min_width - (w + 2*w1)) == 1 else w1
            y_top += h1+1 if (self.min_height - (h + 2*h1)) == 1 else h1
            new_values = np.array([x_left, y_top, bb_width, bb_height])

        if self.plot_extensive:
            print("After padding:", padded_img.shape)
            plt.imshow(padded_img)
        return padded_img, new_values


    def rotation(self, image_path, bbox):
        '''Takes an input image and returns a rotated version of it'''
        # read the image path
        im = cv2.imread(image_path)  # loads image as BGR in uint8
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        h, w = image.shape[:2]
        new_values = None

        # convert to gray image
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # use threshold to find contours of relevant gray image
        ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        # rectangle detected around the relevant image
        # rect = ((center_x,center_y), (width,height), angle)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # get width and height of the detected rectangle
        rect_width = int(rect[1][0])
        rect_height = int(rect[1][1])

        src_pts = box.astype("float32")
        if rect_height < rect_width:
            # coordinate of the points in box points after the rectangle has been straightened
            dst_pts = np.array([[rect_height-1, rect_width-1],
                                [0, rect_width-1],
                                [0, 0],
                                [rect_height-1, 0]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(image, M, (rect_height, rect_width))

        else:
            # coordinate of the points in box points after the rectangle has been straightened
            dst_pts = np.array([[0, rect_height-1],
                                [0, 0],
                                [rect_width-1, 0],
                                [rect_width-1, rect_height-1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(image, M, (rect_width, rect_height))
        warped = warped.astype(np.float32)

        # look for bounding box
        if not (bbox == 0).all():
            # read dimensions of bbox_eye
            cX = (int)(bbox[0])
            cY = (int)(bbox[1])
            bb_width = (int)(bbox[2])
            bb_height = (int)(bbox[3])

            # recompute center of bbox according to transformation matrix
            new_cX = (M[0][0]*cX + M[0][1]*cY + M[0][2]) / ((M[2][0]*cX + M[2][1]*cY + M[2][2]))
            new_cY = (M[1][0]*cX + M[1][1]*cY + M[1][2]) / ((M[2][0]*cX + M[2][1]*cY + M[2][2]))
            new_values = np.array([new_cX, new_cY, bb_width, bb_height])

            if self.plot_extensive:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(image)
                ax1.add_patch(matplotlib.patches.Rectangle((cX, cY), bb_width, bb_height, color='green', fc='none'))
                ax1.set(title="Original image with bbox (green)")
                ax2.imshow(warped)
                ax2.add_patch(matplotlib.patches.Rectangle((new_cX, new_cY), bb_width, bb_height, color='green', fc='none'))
                ax2.set(title="Rotated with bbox (green)")
                fig.tight_layout()
                plt.show()

        if self.plot_extensive:
            print("After rotating:", warped.shape)
            plt.imshow(warped)
        return warped, new_values


    def random_crop(self, image, new_bbox_values):
        '''Takes an input image and returns a cropped version of it'''
        height, width = image.shape[:2]
        if height <= self.min_height and width <= self.min_width:
            return image, new_bbox_values

        new_values = None
        if new_bbox_values is None:
            # there is no eye, select random crop with dimensions (MIN_HEIGHT, MIN_WIDTH)
            box_to_crop = self.select_crop(image, eye=False)
            (x, y, w, h) = box_to_crop

        else:
            # read dimensions of bbox_eye
            x_left = (int)(new_bbox_values[0])
            y_top = (int)(new_bbox_values[1])
            bb_width = (int)(new_bbox_values[2])
            bb_height = (int)(new_bbox_values[3])
            bbox_eye = (x_left, y_top, bb_width, bb_height)

            # select random crop with dimensions (MIN_HEIGHT, MIN_WIDTH) containing eye in bbox_eye
            box_to_crop = self.select_crop(image, bbox_eye, eye=True)
            (x, y, w, h) = box_to_crop

            # recompute and normalize the x1 and y1 according to new dimensions of crop
            new_x_left = (x_left - x)
            new_y_top = (y_top - y)
            new_values = np.array([new_x_left, new_y_top, bb_width, bb_height])

        # crop image with the random crop
        img_cropped = image[int(y):int(y+h), int(x):int(x+w)]

        if self.plot_extensive:
            print("After cropping:", img_cropped.shape)
            plt.imshow(img_cropped)
        assert img_cropped.shape[0] == self.min_height
        assert img_cropped.shape[1] == self.min_width
        return img_cropped, new_values

    #######################################
    ######      HELPER FUNCTIONS      #####
    #######################################

    def define_search_box(self, img):
        """ Defines the box where to search for a random point. This box 
        is computed to prevent additional padding in case the random point
        would appear next to the borders of the image. """
        (h1, w1) = img.shape[:2]
        x = self.min_width/2
        y = self.min_height/2
        w = w1 - self.min_width
        h = h1 - self.min_height
        return (x, y, w, h)

    def pick_random_pt(self, bbox):
        """ Collects a random point from inside the box. """
        (x_left, y_top, w, h) = bbox
        x_right = x_left + w
        y_bottom = y_top + h
        pnt = Point(random.uniform(x_left, x_right), random.uniform(y_top, y_bottom))
        return pnt

    def compute_intersection(self, bb1, bb2):
        """ Calculates the overlap of two bounding boxes. """
        (x1, y1, w1, h1) = bb1
        (x2, y2, w2, h2) = bb2

        # determine the coordinates of the intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1+w1, x2+w2)
        y_bottom = min(y1+h1, y2+h2)
        intersection_rect = (x_left, y_top, x_right-x_left, y_bottom-y_top)

        if x_right < x_left or y_bottom < y_top:
            overlap = 0.0
            return overlap, intersection_rect

        # compute the intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = w1 * h1
        bb2_area = w2 * h2

        # compute the overlap
        overlap = intersection_area / bb1_area
        return overlap, intersection_rect

    def select_crop(self, im, bbox1=None, eye=True):
        """ Selects the random crop with the dimensions (MIN_WIDTH, MIN_HEIGHT), 
        knowing it has to contain the bounding box around the eye (bbox1). If 
        there is no eye in the image, the crop is totally random. """

        # define box where to search for a random point
        search_box = self.define_search_box(im)
        (x, y, w, h) = search_box

        if eye:
            # get dimensions from the bounding box around the eye
            (x1, y1, w1, h1) = bbox1

            (height, width) = im.shape[:2]
            box_img = (0, 0, width, height)
            max_overlap, _ = self.compute_intersection(bbox1, box_img)

            # only stop if bounding boxes are completely overlapped
            overlap = 0
            max_overlap_rounded = (math.floor(max_overlap*100)/100) - 0.005
            while overlap < max_overlap_rounded:
                # pick random point from inside search box
                pnt = self.pick_random_pt(search_box)

                # define a box centered in pnt with the dimensions (MIN_WIDTH, MIN_HEIGHT)
                bbox2 = (pnt.x-self.min_width/2, pnt.y-self.min_height/2, self.min_width, self.min_height)
                (x2, y2, w2, h2) = bbox2

                # compute the overlap between bounding boxes
                overlap, intersection_rect = self.compute_intersection(bbox1, bbox2)
        else:
            # pick random point from inside search box
            pnt = self.pick_random_pt(search_box)

            # define a box centered in pnt with the dimensions (MIN_WIDTH, MIN_HEIGHT)
            bbox2 = (pnt.x-self.min_width/2, pnt.y-self.min_height/2, self.min_width, self.min_height)
            (x2, y2, w2, h2) = bbox2

        if self.plot_light:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(im)
            plt.title("Original image, search box (green) and crop (blue)")
            ax.add_patch(matplotlib.patches.Rectangle((x, y), w, h, color='green', fc='none'))
            if eye:
                ax.add_patch(matplotlib.patches.Rectangle((x1, y1), w1, h1, color='red', fc='none'))
            ax.add_patch(matplotlib.patches.Rectangle((x2, y2), w2, h2, color='blue', fc='none'))
            ax.autoscale()
            plt.show()
        return bbox2