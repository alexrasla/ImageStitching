from turtle import shape
import cv2
import argparse
import os
import natsort
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--images")
args = parser.parse_args()

def find_keypoints(images_path):
    
    matching_keypoints = []
    
    images = os.listdir(images_path)
    images = natsort.natsorted(images) #sorts images by number
    
    for index in range(0, len(images) - 1):

        img_keypoints = []
        img1_path = os.path.join(images_path, images[index])
        img2_path = os.path.join(images_path, images[index+1])
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
       
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        for match in matches:
            p1 = kp1[match.queryIdx].pt
            p2 = kp2[match.trainIdx].pt
            
            img_keypoints.append((p1, p2))
            
        # match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
        # cv2.imshow("img", match_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        matching_keypoints.append(img_keypoints)
    
    return matching_keypoints

def calculate_homography_matrix(keypoints):
 
    for keypoint_pairs in keypoints:
        # homography matrix computed for each img1, img2 pair
        A = np.zeros(shape=(len(keypoint_pairs) + 1,9))
        b = np.zeros(shape=(len(keypoint_pairs) + 1))
        
        for index, pair in enumerate(keypoint_pairs):
            # print(pair)
            x1 = pair[0][0]
            y1 = pair[0][1]
            x2 = pair[1][0]
            y2 = pair[1][1]
            
            A[index] = [x1, y1, 1, 0, 0, 0, -1*x2*x1, -1*x2*y1, -1*x2]
            A[index+1] = [0, 0, 0, x1, y1, 1, -1*y2*x1, -1*y2*y1, -1*y2]
        
        
        w, v = np.linalg.eig(A.T @ A) 
    
    index = np.where(w == np.amin(w))
    homography = v[index]
    
    return np.reshape(homography, (3, 3))
    

if __name__ == "__main__":
    images_path = args.images
    
    matching_keypoints = find_keypoints(images_path)
    homography = calculate_homography_matrix(matching_keypoints)
    
    print(homography)
    

    
    
    