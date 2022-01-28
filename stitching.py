import math
from nis import match
from matplotlib.path import Path
import cv2
import argparse
import os
from matplotlib import pyplot as plt
import natsort
import numpy as np
from shapely.geometry import Polygon, Point
import time

parser = argparse.ArgumentParser()
parser.add_argument("--images")
args = parser.parse_args()

def find_keypoints(src_img, dest_img):
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(dest_img, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    keypoints = []
    for match in matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        
        keypoints.append([p1, p2])

    # img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:50], img1)
    # plt.imshow(img3)
    # plt.show()
    keypoints = np.array([np.array(xi) for xi in keypoints])
    
    return keypoints

def find_homography_matrix(keypoints, iterations=100, epsilon=4):
    '''
    Finds the best homography matrix with RANSAC outlier detection
    '''

    #RANSAC
    best_homography = np.zeros((3, 3))
    best_num_inliers = 0
    for it in range(iterations):
        rand_indexes = np.random.choice(keypoints.shape[0], size=4)  
        rand_keypoints = keypoints[rand_indexes]
        
        homography = calculate_homography_matrix(rand_keypoints)
        
        #get number of inliers
        num_inliers = 0
        for index in range(0, len(keypoints)):
            pair = keypoints[index]
            x2 = pair[1][0]
            y2 = pair[1][1]
            
            src_pixels = np.array([pair[0][0], pair[0][1], 1])
            dest_pixels = homography @ src_pixels
            
            if abs(x2 - dest_pixels[0]/dest_pixels[2]) <= epsilon and abs(y2 - dest_pixels[1]/dest_pixels[2]):
                num_inliers += 1
        
        #update best inliers and homography
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_homography = homography
            
    return best_homography
    
def calculate_homography_matrix(keypoints):
    
    A = np.zeros(shape=(2*len(keypoints),9))
    
    for index in range(0, len(keypoints)):
        pair = keypoints[index]
        x1 = pair[0][0]
        y1 = pair[0][1]
        x2 = pair[1][0]
        y2 = pair[1][1]
        
        # print(pair)
        A[2*index] = [x1, y1, 1, 0, 0, 0, -1*x2*x1, -1*x2*y1, -1*x2]
        A[(2*index) + 1] = [0, 0, 0, x1, y1, 1, -1*y2*x1, -1*y2*y1, -1*y2]
    
    u, s, vh = np.linalg.svd(A)
    homography = vh[len(vh) - 1]
    
    homography = vh[len(vh) - 1]
    homography = np.reshape(homography, (3, 3))
    homography = np.divide(homography, homography[2][2])

    return np.reshape(homography, (3, 3))
    
def warp_images(src_img_path, des_img_path, des_homography, prev_homographies, previous_y_shift, pano_path):
   
    img1 = cv2.imread(src_img_path)
    img2 = cv2.imread(des_img_path)
    
    img1_y_bound, img1_x_bound, _ = img1.shape
    img2_y_bound, img2_x_bound, _ = img2.shape
    
    if len(prev_homographies) > 0:
        src_homography = np.identity(3)
        for hmg in prev_homographies:
            src_homography = src_homography  @ hmg
        src_homography = src_homography @ des_homography
        src_polygon, src_boundaries = get_polygon(src_homography, img1_x_bound, img1_y_bound) 
        des_polygon, des_boundaries = get_polygon(des_homography, img2_x_bound, img2_y_bound)
    else:
        src_homography = des_homography
        src_polygon, src_boundaries = get_polygon(src_homography, img1_x_bound, img1_y_bound) 
        des_polygon, des_boundaries = get_polygon(None, img2_x_bound, img2_y_bound)
    
    
    xs, ys = src_boundaries[:, 0], src_boundaries[:, 1]
    xd, yd = des_boundaries[:, 0], des_boundaries[:, 1]
        
    y_img_min = round(np.min(ys))
    y_img_max = round(np.max(ys))
    
    x_img_min = round(np.min(xs))
    x_img_max = round(np.max(xs))
    
    des_img_height = y_img_max - y_img_min
    des_img_width  = x_img_max - x_img_min
    
    #copy old panoramic to new one
    if pano_path == '':        
        pano_start = abs(y_img_min)
        pano_end = img2.shape[0] + pano_start
                
        pano_image = np.zeros((y_img_max - y_img_min, x_img_max, 3), np.uint8)
        pano_image[pano_start:pano_end, 0:img2.shape[1]] = img2[0:img2.shape[0], 0:img2.shape[1]]
        
    else:
        prev_pano_image = cv2.imread(pano_img_path)
        pano_image = np.zeros((y_img_max - y_img_min, x_img_max, 3), np.uint8)
        
        pano_start = abs(y_img_min - previous_y_shift) 
        pano_end = prev_pano_image.shape[0] + pano_start
        
        prev_width_end = prev_pano_image.shape[1]
        prev_height_end = prev_pano_image.shape[0]
                        
        pano_image[pano_start:pano_end, 0:prev_width_end] = prev_pano_image[0:prev_height_end, 0:prev_width_end]

        # plt.imshow(pano_image)
        # plt.show()
        
    x, y = np.meshgrid(np.arange(x_img_min, x_img_max), np.arange(y_img_min, y_img_max)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
        
    grid = src_polygon.contains_points(points)
    mask = grid.reshape(des_img_height, des_img_width).T # now you have a mask with points inside a polygon
    
    temp_src_points = np.argwhere(mask)
    
    src_points = np.empty((3, temp_src_points.shape[0]), dtype=int)
    src_points[0] = np.add(temp_src_points[:, 0], x_img_min) #shift the homography image to new place
    src_points[1] = np.add(temp_src_points[:, 1], y_img_min) 
    src_points[2] = np.ones(temp_src_points.shape[0])
    
    # plt.figure()
    # plt.gca().invert_yaxis()
    # plt.plot(xs,ys) 
    # plt.plot(xd,yd)
    # plt.scatter(src_points[0], src_points[1]) 
    # plt.show()
    # print('points', src_points)
    
    src_inv_homography = np.linalg.inv(src_homography)    

    src_coords = src_inv_homography @ src_points
    src_coords = np.divide(src_coords, src_coords[2]).astype(int)    
    
    # print('points', np.min(np.add(src_points[1, :], abs(y_img_min))), np.min(src_points[0, :]))
    # print('points max', np.max(np.add(src_points[1, :], abs(y_img_min))), np.max(src_points[0, :]))
    # print('coords', np.min(src_coords[1, :]), np.min(src_coords[0, :]))
    # print('pano dim', pano_image.shape)
    # print('shift by', y_img_min)
    
    
    try:
        pano_image[np.add(src_points[1, :], abs(y_img_min)), src_points[0, :]] = img1[src_coords[1, :], src_coords[0, :]]
    except:
        print("FAILED")
        plt.imshow(pano_image)
        plt.show()
        raise
    return pano_image, y_img_min
    
def get_polygon(homography, x_bound, y_bound):

    if type(homography) == type(None):
        return Path([[0, 0], [x_bound, 0], [x_bound, y_bound], [0, y_bound], [0, 0]]), np.array(([[0, 0], [x_bound, 0], [x_bound, y_bound], [0, y_bound], [0, 0]]))
    
    ul = homography @ np.array([0,0,1])
    ur = homography @ np.array([x_bound,0,1])
    ll = homography @ np.array([0, y_bound,1])
    lr = homography @ np.array([x_bound,y_bound,1])
    
    ul = np.divide(ul, ul[2])
    ur = np.divide(ur, ur[2])
    ll = np.divide(ll, ll[2])
    lr = np.divide(lr, lr[2])
    
    polygon = Path([ul[:2], ur[:2], lr[:2], ll[:2], ul[:2]])

    return polygon, np.array([ul[:2], ur[:2], lr[:2], ll[:2], ul[:2]])
    
if __name__ == "__main__":
    images_path = args.images
    pano_image_path = 'pano_img.jpg'
    
    images = os.listdir(images_path)
    images = natsort.natsorted(images) #sorts images by number
    
    pano_img_path = os.path.join(images_path, pano_image_path)
    
    previous_homographies = []
    previous_y_shift = 0
    
    for index in range(1, len(images) - 1):

        src_img_path = os.path.join(images_path, images[index])
        des_img_path = os.path.join(images_path, images[index - 1])
        
        src_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
        des_img = cv2.imread(des_img_path, cv2.IMREAD_GRAYSCALE)
        
        matching_keypoints = find_keypoints(src_img, des_img)
        
        iterations = math.ceil(math.log(1-0.99)/ math.log(1- (1-0.7)**4))
        homography = find_homography_matrix(matching_keypoints, iterations=iterations)
        
        if index == 1:
            pano_image, previous_y_shift = warp_images(src_img_path, des_img_path, homography, previous_homographies, previous_y_shift, pano_path='')
        else:
            pano_image, previous_y_shift = warp_images(src_img_path, des_img_path, homography, previous_homographies, previous_y_shift, pano_path=pano_img_path)
        
        cv2.imwrite(pano_img_path, pano_image)    
        # cv2.imshow("image", pano_image)
        # cv2.waitKey(0)
        
        previous_homographies.append(homography)
        print(index)
    
    print("Panoramic Image saved at", pano_img_path)
        
        
        


    
    
    