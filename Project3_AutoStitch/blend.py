import math
import sys

import cv2
import numpy as np
from numpy.linalg import inv


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    def getTransformed (point, M):
        transform = np.dot(M,point)
        return (transform[0]/transform[2],transform[1]/transform[2])
    #topleft, top right, bot left ,bot right
    edges = [np.array([0,0,1]), np.array([0,img.shape[0]-1,1]),np.array([img.shape[1]-1, 0,1])\
        , np.array([img.shape[1]-1, img.shape[0]-1,1])]
    x_cords =[]
    y_cords=[]
    for edge in edges:
        t= getTransformed(edge,M)
        x_cords.append(t[0])
        y_cords.append(t[1])
    
    minX = min(x_cords)
    minY = min(y_cords)
    maxX = max(x_cords)
    maxY = max(y_cords)
    
    #print(img.shape)
    #TODO-BLOCK-END
    #print('done')
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    #Minv*acc(x',y')->I(x,y (img))
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    img_h,img_w,channels = img.shape
    accDim = imageBoundingBox(img,M)
    minX, minY, maxX, maxY =  accDim

    for x in range(minX,maxX):
        for y in range(minY,maxY):
            accPt =  np.array([x,y,1.0])
            trans = np.dot(inv(M),accPt)
            newx,newy = trans[0]/trans[2], trans[1]/trans[2]

        if newx >= 0 and newx < img_w-1 and newy >= 0 and newy < img_h-1:    
            weight = 1.0
            if newx >= minX and newx < minX + blendWidth:
                weight = 1. * (newx - minX) / blendWidth
            if newx <= maxX and newx > maxX - blendWidth:
                weight = 1. * (maxX - newx) / blendWidth
            acc[y,x,3] += weight
        
            for k in range(3):
                acc[y,x,k] += img[int(newy),int(newx),k] * weight
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    h_acc = acc.shape[0]
    w_acc = acc.shape[1]
    img = np.zeros((h_acc, w_acc, 3))
    for i in range(0, w_acc, 1):
        for j in range(0, h_acc, 1):
            if acc[j,i,3]>0:
                img[j,i,0] = int (acc[j,i,0] / acc[j,i,3])
                img[j,i,1] = int (acc[j,i,1] / acc[j,i,3])
                img[j,i,2] = int (acc[j,i,2] / acc[j,i,3])
            else:
                img[j,i,0] = 0
                img[j,i,1] = 0
                img[j,i,2] = 0
    img = np.uint8(img)
    return img
    
    


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position  # transform
        img = i.img  # img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        min_x,min_y,max_x,max_y = imageBoundingBox(img,M)
        minX = min(minX,min_x)
        minY = min(minY,min_y)
        maxX = max(maxX,max_x)
        maxY = max(maxY,max_y)
        
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    #print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

