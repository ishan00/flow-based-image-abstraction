import numpy as np

sigc = 1.0
p = 0.9761
r = 0.999
sigm = 3.0
sig_spatial = 2.0
sig_intensity1 = 150.0
sig_intensity2 = 50.0
bins = 8
siz = 256//bins
width = siz//2

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