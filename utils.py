import numpy as np
import cv2

sigc = 1.0
p = 0.9761
r = 0.999
R = [0.4,0.5,0.6,0.7,0.8, 0.9, 0.95]
sigm = 3.0
sig_spatial = 2.0
sig_intensity1 = 150.0
sig_intensity2 = 50.0
bins = 8
BINS = [8,16]

def gaussian(t,sig):
	return (1/np.sqrt(2*np.pi*sig))*np.exp(-t**2/(2*sig**2))


def flow_neighbour(angle):

	angles = [0, 45, 90, 135, 180, 225, 270, 315]
	pairs = [(0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)]

	min_ = 360
	ind = 0
	for i in range(8):
		if abs(angles[i] - angle) < min_:
			min_ = abs(angles[i] - angle)
			ind = i

	return pairs[ind]

def angle(x,y):
	if x == 0:
		return np.pi/2
	else:
		res = np.arctan(abs(y)/abs(x))
		if x >= 0 and y >= 0:
			return 2*np.pi - res
		elif x >= 0 and y < 0:
			return res
		elif x < 0 and y >= 0:
			return np.pi + res
		elif x < 0 and y < 0:
			return np.pi - res	

def dog_filter(t,sigc,p):
	sigs = 1.05 * sigc
	# 1.6 - 0.8116
	# 1.05 - 0.9761
	return gaussian(t,sigc) - p*gaussian(t,sigs)

def intensity_weight(rgb1, rgb2, sigma):
	I = np.dot(rgb1-rgb2,rgb1-rgb2)
	return (1/np.sqrt(2*np.pi*sigma))*np.exp(-I**2/(2*sigma**2))

def color_segmentation(image,batch = False, greyscale = False):

	image1 = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_BGR2LAB)

	size = image.shape

	if batch:

		result = [image1 for _ in range(len(BINS))]

		for i in range(len(BINS)):
			
			siz = 256//BINS[i]
			width = siz//2

			lum = image1[:,:,0]

			for h in range(size[0]):
				for w in range(size[1]):

					diff = lum[h][w] % siz - width

					lum[h][w] = siz*(lum[h][w] // siz) + width + width*np.tanh(diff)

			result[i][:,:,0] = lum

			result[i] = cv2.cvtColor(result[i], cv2.COLOR_LAB2BGR)

		return result

	else:

		siz = 256//bins
		width = siz//2

		lum = image1[:,:,0]

		for h in range(size[0]):
			for w in range(size[1]):

				diff = lum[h][w] % siz - width

				lum[h][w] = siz*(lum[h][w] // siz) + width + width*np.tanh(diff)

		image1[:,:,0] = lum

		if greyscale:
			image1[:,:,1] = np.full((size[0],size[1]),127)
			image1[:,:,2] = np.full((size[0],size[1]),127)

		image2 = cv2.cvtColor(image1, cv2.COLOR_LAB2BGR)

		return image2

def print_progress(prefix, current, total):

	l = len(prefix)

	frac = (current + 1)/total

	frac = (int(100*frac)//5)

	print (prefix + ' '*(15-l) + '[' + '='*frac + '-'*(20-frac) + '] ' + str(5*frac) + '% \r',end='')








