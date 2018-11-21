import cv2
import numpy as np
from numpy import linalg as LA
#import lic_internal
import pylab as plt
import time
from utils import *

def ETF(gray_image, iterations, mu, type):
	gray_image = np.divide(gray_image, 255.0)
	sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
	sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
	sobelmag = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
	sobelm = np.divide(sobelmag, np.amax(sobelmag))
	
	tangx = -1 * sobely
	tangy = sobelx
	tang = np.stack([tangx, tangy], axis=2)
	size = tang.shape
	
	tangnorm = LA.norm(tang, axis=2)
	np.place(tangnorm, tangnorm == 0, [1])
	tang = np.divide(tang, np.stack([tangnorm, tangnorm], axis=2))

	for iteration in range(iterations):
		start = time.time()

		if (type == 0):
			tang_horz = np.zeros(size)

			for h in range(size[0]):
				print_progress('ETF Horiz',h,size[0])
				for w in range(size[1]):
					total_weight = 0.0
					for j in range(max(0, w-mu), min(size[1], w+mu+1)):
						weight = ((sobelm[h][w]-sobelm[h][j]+1)*np.dot(tang[h][w], tang[h][j]))/2.0
						total_weight += weight
						tang_horz[h][w] += tang[h][j]*weight
					if(total_weight != 0):
						tang_horz[h][w] /= total_weight

			tang = np.zeros(size)

			print("")

			for h in range(size[0]):
				print_progress('ETF Vert',h,size[0])
				for w in range(size[1]):
					total_weight = 0.0
					for i in range(max(0, h-mu), min(size[0], h+mu+1)):
						weight = ((sobelm[h][w]-sobelm[i][w]+1)*np.dot(tang_horz[h][w], tang_horz[i][w]))/2.0
						total_weight += weight
						tang[h][w] += tang_horz[i][w]*weight
					if(total_weight != 0):
						tang[h][w] /= total_weight

			print("")

		elif (type == 1):
			tang_new = np.zeros(size)

			for h in range(size[0]):
				if((h % 20) == 0):
					print('h', h//20)
				for w in range(size[1]):
					total_weight = 0.0
					for i in range(max(0, h-mu), min(size[0], h+mu+1)):
						for j in range(max(0, w-mu), min(size[1], w+mu+1)):
							weight = ((sobelm[h][w]-sobelm[i][j]+1)*np.dot(tang[h][w], tang[i][j]))/2.0
							total_weight += weight
							tang_new[h][w] += tang[i][j]*weight
					if(total_weight != 0):
						tang_new[h][w] /= total_weight

			tang = tang_new

		end = time.time()
		#print(end-start,'seconds')

	tangnorm = LA.norm(tang, axis=2)
	np.place(tangnorm, tangnorm == 0, [1])
	tang = np.divide(tang, np.stack([tangnorm, tangnorm], axis=2))

	# texture = np.random.rand(size[0],size[1]).astype(np.float32)
	# kernellen=31
	# kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
	# kernel = kernel.astype(np.float32)

	# for h in range(size[0]):
	# 	for w in range(size[1]):
	# 		if(tang[h][w][0] == 0):
	# 			tang[h][w][0] = np.random.rand()*0.01
	# 		if(tang[h][w][1] == 0):
	# 			tang[h][w][1] = np.random.rand()*0.01

	# tangnorm = LA.norm(tang, axis=2)
	# np.place(tangnorm, tangnorm == 0, [1])
	# tang = np.divide(tang, np.stack([tangnorm, tangnorm], axis=2))

	# etf_lic = lic_internal.line_integral_convolution(tang.astype(np.float32), texture, kernel)

	# dpi = 100
	# plt.bone()
	# plt.clf()
	# plt.axis('off')
	# plt.figimage(etf_lic)
	# plt.gcf().set_size_inches((size[0]/float(dpi),size[1]/float(dpi)))
	# plt.savefig("old_etf.png",dpi=dpi)

	return tang