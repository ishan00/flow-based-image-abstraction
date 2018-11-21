from utils import *
import numpy as np

def FBL(image, etf, iterations = 1):
	
	size = image.shape

	H4 = image

	for i in range(iterations):

		H3 = np.zeros(size)

		for h in range(size[0]):
			print ("FBL S " + str(100.0*h/size[0]) + "\r")
			for w in range(size[1]):
				total_weight = 0
				ang = angle(etf[h][w][0],etf[h][w][1])
				pix = flow_neighbour(ang*180/np.pi)
				for j in range(-5,6):
					if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue
					H3[h][w] += H4[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sig_spatial)*intensity_weight(H4[h][w],H4[h+pix[0]*j][w+pix[1]*j],sig_intensity1)
					total_weight += gaussian(abs(j),sig_spatial)*intensity_weight(H4[h][w],H4[h+pix[0]*j][w+pix[1]*j],sig_intensity1)
					if h == 60 and w == 60:
						print(j,gaussian(abs(j),sig_spatial)*intensity_weight(H4[h][w],H4[h+pix[0]*j][w+pix[1]*j],sig_intensity1))
				H3[h][w] /= total_weight
		
		print ("")
		H4 = np.zeros(size)

		for h in range(size[0]):
			print ("FBL T " + str(100.0*h/size[0]) + "\r")
			for w in range(size[1]):
				total_weight = 0
				angle_perpendicular = (angle(etf[h][w][0],etf[h][w][1]) + np.pi/2)
				
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

		print("")
	return H4