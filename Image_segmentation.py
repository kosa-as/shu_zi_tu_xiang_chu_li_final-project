import os
import sys
import cv2
import numpy as np
import random as rng


def main():

        img = cv2.imread('./image-se.jpg')

        src = img

        img[np.all(img == 255, axis=2)] = 0

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

        Laplacian = cv2.filter2D(img, cv2.CV_32F, kernel)

        imgres = np.float32(img) - Laplacian

        imgres = np.clip(imgres, 0, 255)

        imgres = np.uint8(imgres)

        bw = cv2.cvtColor(imgres, cv2.COLOR_BGR2GRAY)

        _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)

        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

        kernel1 = np.ones((3, 3), dtype=np.uint8)

        dist = cv2.dilate(dist, kernel1)

        dist_8u = dist.astype('uint8')

        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = np.zeros(dist.shape, dtype=np.int32)

        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i + 1), -1)

        cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)

        cv2.watershed(imgres, markers)

        mark = markers.astype('uint8')

        colors = []

        for conlour in contours:
            colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

        dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i, j]
                if index > 0 and index <= len(contours):
                    dst[i, j, :] = colors[index - 1]

        img = np.hstack([src, dst])

        cv2.imshow('Final Result', img)

        cv2.waitKey(0)