import cv2
import numpy as np
from ETF import ETF
from FDoG import FDoG
from FBL import FBL
from utils import *
# from scipy.io import savemat

def main(input_path, output_path, flags):

	image = cv2.imread(input_path, 1)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	etf = ETF(gray_image, iterations = 1, mu = 5, type = 0)

	edges = FDoG(gray_image, etf, iterations = 2)
	smoothed_image = FBL(image, etf, iterations = 1)

	size = image.shape

	image2 = cv2.cvtColor(smoothed_image,cv2.COLOR_BGR2LAB)

	lum = image2[:,:,0]

	for h in range(size[0]):
		for w in range(size[1]):

			diff = lum[h][w] % siz - width

			lum[h][w] = siz*(lum[h][w] // siz) + width + width*np.tanh(diff)

	image2[:,:,0] = lum

	print (np.amax(image2[:,:,0]), np.amin(image2[:,:,0]))

	image3 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)

	image3 = np.minimum(image3,np.stack([edges, edges, edges], axis=2))

	cv2.imwrite(output_path,image3)

