import cv2
import numpy as np
from ETF import ETF
# from scipy.io import savemat

image = cv2.imread('macaw.jpg', 1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

etf = ETF(gray_image, iterations=1, mu=5, type=0)

sigc = 1.0
p = 0.9761
r = 0.999
sigm = 3.0

lines = FDoG(gray_image, etf)



# def gaussian(t,sig):
# 	return (1/np.sqrt(2*np.pi*sig))*np.exp(-t**2/(2*sig**2))

# def dog_filter(t,sigc,p):
# 	sigs = 1.05 * sigc
# 	# 1.6 - 0.8116
# 	# 1.05 - 0.9761
# 	return gaussian(t,sigc) - p*gaussian(t,sigs)

# def flow_neighbour(angle):

# 	angles = [0, 45, 90, 135, 180, 225, 270, 315]
# 	pairs = [(0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)]

# 	min_ = 360
# 	ind = 0
# 	for i in range(8):
# 		if abs(angles[i] - angle) < min_:
# 			min_ = abs(angles[i] - angle)
# 			ind = i

# 	return pairs[ind]

# def angle(x,y):
# 	if x == 0:
# 		return np.pi/2
# 	else:
# 		res = np.arctan(abs(y)/abs(x))
# 		if x >= 0 and y >= 0:
# 			return 2*np.pi - res
# 		elif x >= 0 and y < 0:
# 			return res
# 		elif x < 0 and y >= 0:
# 			return np.pi + res
# 		elif x < 0 and y < 0:
# 			return np.pi - res		

# H1 = np.zeros((size[0],size[1]))
# H2 = np.zeros((size[0],size[1]))

# for h in range(size[0]):
# 	print ('dog',h)
# 	for w in range(size[1]):

# 		total_weight = 0
# 		angle_perpendicular = (angle(tang[h][w][0],tang[h][w][1]) + np.pi/2)
		
# 		if angle_perpendicular > 2*np.pi:
# 			angle_perpendicular -= 2*np.pi

# 		pix = flow_neighbour(angle_perpendicular*180/np.pi)

# 		for j in range(-3,4):

# 			if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue 

# 			H1[h][w] += image[h+pix[0]*j][w+pix[1]*j]*dog_filter(abs(j),sigc,p)
# 			if h == 60 and w == 60:
# 				print(j,dog_filter(abs(j),sigc,p))

# for h in range(size[0]):
# 	print('gaussian',h)
# 	for w in range(size[1]):

# 		total_weight = 0
# 		ang = angle(tang[h][w][0],tang[h][w][1])
# 		pix = flow_neighbour(ang*180/np.pi)

# 		for j in range(-3,4):

# 			if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue

# 			H2[h][w] += H1[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sigm)
# 			if h == 60 and w == 60:
# 				print(j,H1[h+pix[0]*j][w+pix[1]*j])

# 			total_weight += gaussian(j,sigm)

# 		H2[h][w] /= total_weight

# ind1 = (H2 < 0).astype(int)
# ind2 = ((1 + np.tanh(H2)) < r).astype(int)

# edges = (1 - np.multiply(ind1,ind2))*255

'''
H1 = H1.tolist()
H2 = H2.tolist()
data = {'H1':H1 , 'H2':H2, 'tang':tang}
savemat('macaw.mat',data)
def intensity_weight(rgb1, rgb2, sigma):
	I = np.dot(rgb1-rgb2,rgb1-rgb2)
	return (1/np.sqrt(2*np.pi*sigma))*np.exp(-I**2/(2*sigma**2))
H3 = np.zeros(size)
H4 = np.zeros(size)
sig_spatial = 2.0
sig_intensity1 = 150.0
sig_intensity2 = 50.0
for h in range(size[0]):
	print('s',h)
	for w in range(size[1]):
		total_weight = 0
		ang = angle(tang[h][w][0],tang[h][w][1])
		pix = flow_neighbour(ang*180/np.pi)
		for j in range(-5,6):
			if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue
			H3[h][w] += image1[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sig_spatial)*intensity_weight(image1[h][w],image1[h+pix[0]*j][w+pix[1]*j],sig_intensity1)
			total_weight += gaussian(abs(j),sig_spatial)*intensity_weight(image1[h][w],image1[h+pix[0]*j][w+pix[1]*j],sig_intensity1)
			if h == 60 and w == 60:
				print(j,gaussian(abs(j),sig_spatial)*intensity_weight(image1[h][w],image1[h+pix[0]*j][w+pix[1]*j],sig_intensity1))
		H3[h][w] /= total_weight
cv2.imwrite('H3.png',H3)
cv2.waitKey(0)
for h in range(size[0]):
	print('t',h)
	for w in range(size[1]):
		total_weight = 0
		angle_perpendicular = (angle(tang[h][w][0],tang[h][w][1]) + np.pi/2)
		
		if angle_perpendicular > 2*np.pi:
			angle_perpendicular -= 2*np.pi
		pix = flow_neighbour(angle_perpendicular*180/np.pi)
		for j in range(-5,6):
			if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue
			H4[h][w] += H3[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sig_spatial)*intensity_weight(H3[h][w],H3[h+pix[0]*j][w+pix[1]*j],sig_intensity2)
			total_weight += gaussian(abs(j),sig_spatial)*intensity_weight(H3[h][w],H3[h+pix[0]*j][w+pix[1]*j],sig_intensity2)
			if h == 60 and w == 60:
				print(j,gaussian(abs(j),sig_spatial)*intensity_weight(H3[h][w],H3[h+pix[0]*j][w+pix[1]*j],sig_intensity2))
		H4[h][w] /= total_weight
cv2.imwrite('H4.png',H4)
cv2.waitKey(0)
'''
'''
H5 = np.zeros(size)
pixel_values = []
for h in range(size[0]):
	for w in range(size[1]):
		pixel_values.append([image1[h][w][0],image1[h][w][1],image1[h][w][2],h,w])
pixel_values = [pixel_values]
def split_list(list):
	rmax = 0.0
	gmax = 0.0
	bmax = 0.0
	rmin = 255.0
	gmin = 255.0
	bmin = 255.0
	for i in range(len(list)):
		rmax = max(rmax, list[i][0])
		gmax = max(gmax, list[i][1])
		bmax = max(bmax, list[i][2])
		rmin = min(rmin, list[i][0])
		gmin = min(gmin, list[i][1])
		bmin = min(bmin, list[i][2])
	rrange = rmax - rmin
	grange = gmax - gmin
	brange = bmax - bmin
	c = -1
	if rrange >= grange and rrange >= brange:
		c = 0
	elif grange >= rrange and grange >= brange:
		c = 1
	else:
		c = 2
	list.sort(key = lambda x : x[c])
	length = len(list)
	return [list[0:length//2], list[length//2:length]]
num = 5
while num > 0:
	temp_list = []
	for sublist in pixel_values:
		ret = split_list(sublist)
		temp_list.append(ret[0])
		temp_list.append(ret[1])
	pixel_values = temp_list
	num -= 1
print (pixel_values)
for sublist in pixel_values:
	rgb = [0.0, 0.0, 0.0]
	for i in range(len(sublist)):
		rgb[0] += sublist[i][0]
		rgb[1] += sublist[i][1]
		rgb[2] += sublist[i][2]
	rgb[0] /= len(sublist)
	rgb[1] /= len(sublist)
	rgb[2] /= len(sublist)
	for i in range(len(sublist)):
		H5[sublist[i][3]][sublist[i][4]] = np.array(rgb)
cv2.imwrite('H5.png',H5)
'''

# image1 = cv2.imread('trump.jpg', 1)

# size = image1.shape

# image2 = cv2.cvtColor(image1,cv2.COLOR_BGR2LAB)

# l = image2[:,:,0]

# print (np.amax(l), np.amin(l))

# bins = 8
# siz = 256//bins
# width = siz//2

# for h in range(size[0]):
# 	for w in range(size[1]):

# 		diff = l[h][w] % siz - width

# 		l[h][w] = siz*(l[h][w] // siz) + width + width*np.tanh(diff)

# image2[:,:,0] = l

# print (np.amax(image2[:,:,0]), np.amin(image2[:,:,0]))

# image3 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)

# image3 = np.minimum(image3,np.stack([edges, edges, edges], axis=2))

# cv2.imwrite('trump_cartoon.png',image3)