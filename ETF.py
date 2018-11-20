import cv2
import numpy as np
from numpy import linalg as LA
#import lic_internal
import pylab as plt
from scipy.io import savemat

dpi = 100
image = cv2.imread('leaf.png', 0)
image = np.divide(image,255.0)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobelmag = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
sobelm = np.divide(sobelmag, np.amax(sobelmag))
print(sobelm.shape)
tangx = -1 * sobely
tangy = sobelx
tang = np.stack([tangx, tangy], axis=2)
size = tang.shape
sigc = 0.5
p = 0.9
r = 0.997
sigm = 1.0

tangnorm = LA.norm(tang, axis=2)
np.place(tangnorm, tangnorm == 0, [1])
tang = np.divide(tang, np.stack([tangnorm, tangnorm], axis=2))

'''
mu = 5
tang_new = np.zeros(size)

print(type(size[0]))
print(tang[200][200])

for h in range(size[0]):
	print(h)
	for w in range(size[1]):
		x = 0
		y = 0
		for i in range(max(0, h-mu), min(size[0], h+mu+1)):
			for j in range(max(0, w-mu), min(size[1], w+mu+1)):
				weight = ((sobelm[h][w]-sobelm[i][j]+1)*np.dot(tang[h][w], tang[i][j]))/2.0
				x += tang[i][j][0]*weight
				y += tang[i][j][1]*weight
		tang_new[h][w][0] = x
		tang_new[h][w][1] = y

tang = tang_new
'''
'''
texture = np.random.rand(size[0],size[1]).astype(np.float32)
kernellen=31
kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
kernel = kernel.astype(np.float32)

tangnorm = LA.norm(tang, axis=2)
np.place(tangnorm, tangnorm == 0, [1])
tang = np.divide(tang, np.stack([tangnorm, tangnorm], axis=2))

'''
'''
for h in range(size[0]):
	for w in range(size[1]):
		if(tang[h][w][0] == 0):
			tang[h][w][0] = np.random.rand()*0.01
		if(tang[h][w][1] == 0):
			tang[h][w][1] = np.random.rand()*0.01

tangnorm = LA.norm(tang, axis=2)
np.place(tangnorm, tangnorm == 0, [1])
tang = np.divide(tang, np.stack([tangnorm, tangnorm], axis=2))

etf_lic = lic_internal.line_integral_convolution(tang.astype(np.float32), texture, kernel)

plt.bone()
plt.clf()
plt.axis('off')
plt.figimage(etf_lic)
plt.gcf().set_size_inches((size[0]/float(dpi),size[1]/float(dpi)))
plt.savefig("res1.png",dpi=dpi)
'''

def gaussian(t,sig):
	return (1/np.sqrt(2*np.pi*sig))*np.exp(-t**2/(2*sig**2))

def dog_filter(t,sigc,p):
	sigs = 1.6 * sigc
	return gaussian(t,sigc) - p*gaussian(t,sigs)

def flow_neighbour(angle):

	angles = [0, 45, 90, 135, 180, 225, 270, 315]
	pairs = [(0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)]

	min_ = 360
	ind = 0
	for i in range(8):
		if abs(angles[i] - angle) < min_:
			min_ = abs(angles[i] - angle)
			ind = i

	return pairs[i]

def angle(x,y):
	if x == 0:
		return np.pi/2
	else:
		res = np.arctan(y/x)
		if res < 0 :
			res += np.pi
		return res

H1 = np.zeros((size[0],size[1]))
H2 = np.zeros((size[0],size[1]))

for h in range(size[0]):
	print ('dog',h)
	for w in range(size[1]):

		total_weight = 0
		angle_perpendicular = (angle(tang[h][w][0],tang[h][w][1]) + np.pi/2)
		
		if angle_perpendicular > 2*np.pi:
			angle_perpendicular -= 2*np.pi

		pix = flow_neighbour(angle_perpendicular*180/np.pi)

		for j in range(-3,4):

			if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue 

			H1[h][w] += image[h+pix[0]*j][w+pix[1]*j]*dog_filter(abs(j),sigc,p)
			total_weight += dog_filter(abs(j),sigc,p)

		H1[h][w] /= total_weight

cv2.imshow('H1',H1)
cv2.waitKey(0)

for h in range(size[0]):
	print('gaussian',h)
	for w in range(size[1]):

		total_weight = 0
		ang = angle(tang[h][w][0],tang[h][w][1])
		pix = flow_neighbour(ang*180/np.pi)

		for j in range(-3,4):

			if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue

			H2[h][w] += H1[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sigm)
			total_weight += gaussian(abs(j),sigm)

		H2[h][w] /= total_weight

cv2.imshow('H2',H2)
cv2.waitKey(0)

H1 = H1.tolist()
H2 = H2.tolist()
data = {'H1':H1 , 'H2':H2}
savemat('data.mat',data)

print (H2)

print (np.amin(H2),np.amax(H2))

ind1 = (H2 < 0.1).astype(int)
#ind2 = (1 + np.tanh(H2) < r).astype(int)

#print (np.sum(np.sum(ind2)))

H3 = (1 - ind1)

cv2.imshow('abc',H3)
cv2.waitKey(0)
