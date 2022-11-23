#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 
import glob


# In[2]:


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# In[3]:


objp=np.zeros((6*7,3),np.float32)

objp[:,:2]=np.mgrid[0:7,0:6].T.reshape(-1,2)
print(objp)


# In[4]:


img = cv2.imread('/Users/jules/Downloads/opencv-camera-calibration/sample/frame-4.png', 0)


# In[5]:


pattern_points = np.zeros((np.prod(54), 3), np.float32)
pattern_points[:, :2] = np.indices(6,9).T.reshape(-1, 2)


# In[6]:


h,w=img.shape[:2]


found,corners=cv2.findChessboardCorners(img,(9,6))
print(found)
if found:
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    cv2.cornerSubPix(img, corners, (5,5), (-1,-1),term)
print(term)


# In[7]:


images = glob.glob('*.jpg')
print(images)


# In[8]:


chessboard_size = (9,6)
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
images=['/Users/jules/Downloads/opencv-camera-calibration/sample/frame-0.png',
       '/Users/jules/Downloads/opencv-camera-calibration/sample/frame-1.png','/Users/jules/Downloads/opencv-camera-calibration/sample/frame-2.png']
folder_dir = "/Users/jules/Downloads/opencv-camera-calibration/sample"
calibration_paths = glob.glob('/Users/jules/Downloads/opencv-camera-calibration/sample/*')
for fname in tqdm(calibration_paths):
    img = cv2.imread(fname)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Image loaded, Analizying...")
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Chessboard detected!")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
       
        #refine corner location (to subpixel accuracy) based on criteria.
        cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
      #  corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
      #  imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

print (len(imgpoints))
print(len(imgpoints))


# In[9]:


ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)


# In[10]:


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret)


# In[11]:


np.save("/Users/jules/Downloads/opencv-camera-calibration/sample/ret", ret)
np.save("/Users/jules/Downloads/opencv-camera-calibration/sample/K", K)
np.save("/Users/jules/Downloads/opencv-camera-calibration/sample/dist", dist)
np.save("/Users/jules/Downloads/opencv-camera-calibration/sample/rvecs", rvecs)
np.save("/Users/jules/Downloads/opencv-camera-calibration/sample/tvecs", tvecs)


# In[45]:


img = cv2.imread(images[0])
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# In[46]:


dst = cv2.undistort(img, mtx, dist, None, newcameramtx)


# In[47]:


x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('/Users/jules/Downloads/opencv-camera-calibration/sample/calibresult.png', dst)


# In[10]:


import sys, json, os
from math import radians
import numpy
import bpy
from mathutils import Matrix

scene = bpy.context.scene
assert scene.render.resolution_percentage == 100


# In[48]:


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )


# In[49]:


from PIL import Image, ExifTags

img = Image.open("/Users/jules/Downloads/opencv-camera-calibration/sample/frame-0.png")
img_exif = img.getexif()
print(type(img_exif))
# <class 'PIL.Image.Exif'>

if img_exif is None:
    print('Sorry, image has no exif data.')


# In[23]:


import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image

#exif_img = Image.open("/Users/jules/Downloads/opencv-camera-calibration/sample/frame-0.png")
exif_img = PIL.Image.open("/Users/jules/Downloads/opencv-camera-calibration/sample/frame-0.png")
img_exif = exif_img.getexif()
exif_data = {
       PIL.ExifTags.TAGS[k]:v
       for k, v in img_exif.items()
       if k in PIL.ExifTags.TAGS}
#Get focal length in tuple form
#for key, val in img_exif.items():
        #if key in ExifTags.TAGS:
            
focal_length_exif = exif_data['FocalLength']

#Get focal length in decimal form
focal_length = focal_length_exif[0]/focal_length_exif[1]
np.save("/Users/jules/Downloads/opencv-camera-calibration/sample/FocalLength", focal_length)


# In[17]:


for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()


# In[ ]:


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# In[ ]:





# In[ ]:





# In[ ]:




