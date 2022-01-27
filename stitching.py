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

# elif des_polygon.contains(Point(i, j)):
                
            #     try:
            #         if np.array_equal(des_homography, src_homography):
            #             pano_image[j-y_img_min, i] = img2[j, i]
            #         else:
            #             des_inv_homography = np.linalg.inv(des_homography)
            #             des_coords = des_inv_homography @ [i, j, 1]
            #             des_coords = np.divide(des_coords, des_coords[2])  

            #             pano_image[j-y_img_min, i] = img2[math.floor(des_coords[1]), math.floor(des_coords[0])]
            #     except Exception as e:
            #         print(e)
            #         print(j-y_img_min, i, j, i)
            #         print(img2.shape)
            #         plt.imshow(pano_image)
            #         plt.show()
            #         raise

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
    
    # if type(previous_homography) == np.ndarray:
    #     for keypoint in keypoints:
    #         new_kp = previous_homography @ np.array([keypoint[1][0], keypoint[1][1], 1])
    #         new_kp = np.divide(new_kp, new_kp[2])
    #         keypoint[1] = np.array([new_kp[0], new_kp[1]])
    
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
    
def warp_images(src_img_path, des_img_path, des_homography, prev_homographies, pano_path):
   
    img1 = cv2.imread(src_img_path)
    img2 = cv2.imread(des_img_path)
    
    img1_y_bound, img1_x_bound, _ = img1.shape
    img2_y_bound, img2_x_bound, _ = img2.shape
    
    if len(prev_homographies) > 0:
        src_homography = np.identity(3)
        for hmg in prev_homographies:
            src_homography = src_homography  @ hmg
        src_homography = src_homography @ des_homography
        src_polygon = get_polygon(src_homography, img1_x_bound, img1_y_bound) 
        des_polygon = get_polygon(des_homography, img2_x_bound, img2_y_bound)
    else:
        src_homography = des_homography
        src_polygon = get_polygon(src_homography, img1_x_bound, img1_y_bound) 
        des_polygon = get_polygon(None, img2_x_bound, img2_y_bound)
    
    # src_polygon = get_polygon(src_homography, img1_x_bound, img1_y_bound) 
    # des_polygon = get_polygon(None, img2_x_bound, img2_y_bound)
    
    xs, ys = src_polygon.exterior.coords.xy
    xd, yd = des_polygon.exterior.coords.xy

    # plt.figure()
    # plt.gca().invert_yaxis()
    # plt.plot(xs,ys) 
    # plt.plot(xd,yd)
    # plt.show()
    
    y_img_min = round(np.min(ys))
    y_img_max = round(np.max(ys))
    
    x_img_min = round(np.min(xs))
    x_img_max = round(np.max(xs))
    
    des_img_height = y_img_max - y_img_min
    des_img_width  = x_img_max - x_img_min
    
    print(des_img_height, des_img_width)
    
    #copy old panoramic to new one
    if pano_path == '':
        pano_image = np.zeros((des_img_height + 1, des_img_width + 1, 3), np.uint8)
        pano_image[0:img2.shape[0], 0:img2.shape[1]] = img2[0:img2.shape[0], 0:img2.shape[1]]
        
        prev_width_end = 0
        prev_height_end = 0
        
    else:
        prev_pano_image = cv2.imread(pano_img_path)
        pano_image = np.zeros((des_img_height + 1, des_img_width + 1, 3), np.uint8)
        
        prev_width_end = prev_pano_image.shape[1]
        prev_height_end = prev_pano_image.shape[0]
        
        try:
            pano_image[0:prev_height_end, 0:prev_width_end] = prev_pano_image[0:prev_height_end, 0:prev_width_end]
        except:
            print("out of range")
            
    src_inv_homography = np.linalg.inv(src_homography)
    
    print('prev', prev_width_end, prev_height_end)
    
    for i in range(des_img_width):
        for j in range(des_img_height):
            if src_polygon.contains(Point(i, j)):
                
                src_coords = src_inv_homography @ [i, j, 1]
                src_coords = np.divide(src_coords, src_coords[2])
                try:
                    pano_image[j, i] = img1[math.floor(src_coords[1]), math.floor(src_coords[0])]
                except Exception as e:
                    print(e)
                    print(j, i, math.floor(src_coords[1]), math.floor(src_coords[0]))
                    print(pano_image.shape,  img2.shape)
                    plt.imshow(pano_image)
                    plt.show()
                    raise
        
    return pano_image
    
def get_polygon(homography, x_bound, y_bound):

    if type(homography) == type(None):
        return Polygon([[0, 0], [x_bound, 0], [x_bound, y_bound], [0, y_bound], [0, 0]])
    
    ul = homography @ np.array([0,0,1])
    ur = homography @ np.array([x_bound,0,1])
    ll = homography @ np.array([0, y_bound,1])
    lr = homography @ np.array([x_bound,y_bound,1])
    
    ul = np.divide(ul, ul[2])
    ur = np.divide(ur, ur[2])
    ll = np.divide(ll, ll[2])
    lr = np.divide(lr, lr[2])
    
    polygon = Polygon([ul[:2], ur[:2], lr[:2], ll[:2], ul[:2]])

    return polygon
    
if __name__ == "__main__":
    images_path = args.images
    pano_image_path = 'pano_img.jpg'
    
    images = os.listdir(images_path)
    images = natsort.natsorted(images) #sorts images by number
    
    # base_img = os.path.join(images_path, images[0])
    previous_homographies = []
    for index in range(1, len(images) - 3):

        src_img_path = os.path.join(images_path, images[index])
        des_img_path = os.path.join(images_path, images[index - 1])
        
        pano_img_path = os.path.join(images_path, pano_image_path)
        
        src_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
        des_img = cv2.imread(des_img_path, cv2.IMREAD_GRAYSCALE)
        
        matching_keypoints = find_keypoints(src_img, des_img)
        
        iterations = math.ceil(math.log(1-0.99)/ math.log(1- (1-0.5)**4))
        homography = find_homography_matrix(matching_keypoints, iterations=iterations)
        
        # homography_arr.append(homography)
        
        # need to multiple homogrpahies 12 23 34 to get 14
        if index == 1:
            pano_image = warp_images(src_img_path, des_img_path, homography, previous_homographies, pano_path='')
        else:
            pano_image = warp_images(src_img_path, des_img_path, homography, previous_homographies, pano_path=pano_img_path)
        #     pano_image = warp_images(src_img_path, pano_img_path, homography, previous_homographies)
        
        cv2.imwrite(pano_img_path, pano_image)    
        cv2.imshow("image", pano_image)
        cv2.waitKey(0)
        
        previous_homographies.append(homography)
    
    print("Panoramic Image saved at", src_img_path)
        
        
        


    
    
    