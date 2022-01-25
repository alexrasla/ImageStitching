import math
from nis import match
import cv2
import argparse
import os
from matplotlib import pyplot as plt
import natsort
import numpy as np
from shapely.geometry import Polygon, Point


parser = argparse.ArgumentParser()
parser.add_argument("--images")
args = parser.parse_args()

def find_keypoints(img1_path, img2_path):
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    keypoints = []
    for match in matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        
        keypoints.append([p1, p2])

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
        
    # print(best_num_inliers, homography)  
    
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

    return np.reshape(homography, (3, 3))
    
def warp_images(img1_path, img2_path, homography):
    # warp img1 onto img2
    # get bounding box of src_img by doing forward homography
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_y_bound, img1_x_bound, _ = img1.shape
    img2_y_bound, img2_x_bound, _ = img2.shape
    
    ul = homography @ np.array([0,0,1])
    ur = homography @ np.array([img1_x_bound,0,1])
    ll = homography @ np.array([0,img1_y_bound,1])
    lr = homography @ np.array([img1_x_bound,img1_y_bound,1])
    
    ul = np.divide(ul, ul[2])
    ur = np.divide(ur, ur[2])
    ll = np.divide(ll, ll[2])
    lr = np.divide(lr, lr[2])
    
    # src_bounding_box = np.array([ul[:2], ur[:2], lr[:2], ll[:2], ul[:2]])
    src_polygon = Polygon([ul[:2], ur[:2], lr[:2], ll[:2], ul[:2]])

    # des_bounding_box = np.array([[0, 0], [0, img1_y_bound], [img1_x_bound, img2_y_bound], [img1_x_bound, 0], [0,0]])
    des_polygon = Polygon([[0, 0], [0, img1_y_bound], [img1_x_bound, img2_y_bound], [img1_x_bound, 0], [0,0]])
    
    des_img_height = math.ceil(np.max([abs(ul[1]-ll[1]), abs(ur[1]-lr[1]), abs(img2_y_bound-ll[1]), abs(img2_y_bound-ur[1]), img2_y_bound]))
    des_img_width = math.ceil(np.max([abs(ul[0]-ll[0]), abs(ur[0]-lr[0]), abs(img2_x_bound-ll[0]), abs(img2_x_bound-ur[0]), img2_x_bound]))
    
    pano_image = np.zeros((des_img_height, des_img_width, 3), np.uint8)
    # xs, ys = zip(*src_bounding_box) #create lists of x and y values
    # xd, yd = zip(*des_bounding_box)

    # plt.figure()
    # plt.plot(xs,ys) 
    # plt.plot(xd, yd)
    # plt.show()
    inv_homography = np.linalg.inv(homography)
    for i in range(des_img_height):
        for j in range(des_img_width):
            if src_polygon.contains(Point(i, j)):
                src_coords = inv_homography @ [i, j, 1]
                src_coords = np.divide(src_coords, src_coords[2])
                pano_image[j, i] = img1[math.floor(src_coords[1]), math.floor(src_coords[0])]
            elif des_polygon.contains(Point(i, j)):
                pano_image[j, i] = img2[j, i]
                
    return pano_image
    

if __name__ == "__main__":
    images_path = args.images
    
    images = os.listdir(images_path)
    images = natsort.natsorted(images) #sorts images by number
    
    src_img_path = os.path.join(images_path, images[0])
    
    for index in range(0, len(images) - 1):
        
        des_img_path = os.path.join(images_path, images[index+1])
        
        matching_keypoints = find_keypoints(src_img_path, des_img_path)
        
        iterations = math.ceil(math.log(1-0.99)/ math.log(1- (1-0.5)**4))
        homography = find_homography_matrix(matching_keypoints, iterations=iterations)
        
        pano_image = warp_images(src_img_path, des_img_path, homography)
        cv2.imwrite(os.path.join(images_path, 'pano_img.jpg'), pano_image)    
        cv2.imshow("image", pano_image)
        
        src_img_path = os.path.join(images_path, 'pano_img.jpg')
    
    print("Panoramic Image saved at", src_img_path)
        
        
        

    
    
    