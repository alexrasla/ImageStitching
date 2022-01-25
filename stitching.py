import math
from nis import match
import cv2
import argparse
import os
from matplotlib import pyplot as plt
import natsort
import numpy as np

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
    
    # match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    # cv2.imshow("img", match_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

    # homography matrix computed for each img1, img2 pair
    # print(keypoints)
    # h, status = cv2.findHomography(keypoints[:, 0], keypoints[:,1])
    # print('h', h)

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
    #warp img1 onto img2
    #get bounding box of src_img by doing forward homography
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    y_bound, x_bound = img1.shape
    # im_dst = cv2.warpPerspective(img2, homography, img2.shape)
    # cv2.imshow("Destination Image", im_dst)
    # cv2.imshow("src", img1)
    # cv2.waitKey(0)

    ul = homography @ np.array([0,0,1])
    ur = homography @ np.array([x_bound,0,1])
    ll = homography @ np.array([0,y_bound,1])
    lr = homography @ np.array([x_bound,y_bound,1])
    
    bounding_box = np.array([ul, ur, ll, lr])
    
    # lables = ['ul', 'ur', 'll', 'lr']
    # print(bounding_box)
    # cv2.drawContours(img1, [ul, ur, ll, lr], -1, (0, 255, 0), 3)

    # image = cv2.polylines(img1, [bounding_box], isClosed=True, color=(255, 0, 0), thickness=2)
    # plt.gca().invert_yaxis()
    # plt.xlim((0, 320))
    plt.scatter(np.divide(bounding_box[:, 0], bounding_box[:, 2]), np.divide(bounding_box[:, 1], bounding_box[:, 2]))
    # for i in range(len(lables)):
    #     plt.annotate(lables[i], (bounding_box[i, 0], bounding_box[i, 1]))
    plt.show()

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    # for i in range(rows):
    #     for j in range(cols):
            
    # dest_img = cv2.

if __name__ == "__main__":
    images_path = args.images
    
    images = os.listdir(images_path)
    images = natsort.natsorted(images) #sorts images by number
    
    for index in range(0, len(images) - 1):
        
        img1_path = os.path.join(images_path, images[index])
        img2_path = os.path.join(images_path, images[index+1])
        
        matching_keypoints = find_keypoints(img1_path, img2_path)
        
        iterations = math.ceil(math.log(1-0.99)/ math.log(1- (1-0.8)**4))
        # print(iterations)
        homography = find_homography_matrix(matching_keypoints, iterations=iterations)
        
        # point = np.array([matching_keypoints[0][0]])
        # point = np.append(point, [1])
        # h = homography @ point
        # print(h/point[2])
        warp_images(img1_path, img2_path, homography)
        
        
        

    
    
    