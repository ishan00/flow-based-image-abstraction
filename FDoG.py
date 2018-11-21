from utils import *
import numpy as np

def FDoG(image, etf, iterations = 1):

	size = image.shape

	H2 = image

	for i in range(iterations):

		H1 = np.zeros(size)

		for h in range(size[0]):
			print ("DOG " + str(100.0*h/size[0]) + "\r")
			for w in range(size[1]):

				total_weight = 0
				angle_perpendicular = (angle(etf[h][w][0],etf[h][w][1]) + np.pi/2)
				
				if angle_perpendicular > 2*np.pi:
					angle_perpendicular -= 2*np.pi

				pix = flow_neighbour(angle_perpendicular*180/np.pi)

				for j in range(-3,4):

					if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue 

					H1[h][w] += H2[h+pix[0]*j][w+pix[1]*j]*dog_filter(abs(j),sigc,p)
					if h == 60 and w == 60:
						print(j,dog_filter(abs(j),sigc,p))

		print("")

		H2 = np.zeros(size)

		for h in range(size[0]):
			print ("Gaussian " + str(100.0*h/size[0]) + "\r")
			for w in range(size[1]):

				total_weight = 0
				ang = angle(etf[h][w][0],etf[h][w][1])
				pix = flow_neighbour(ang*180/np.pi)

				for j in range(-3,4):

					if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue

					H2[h][w] += H1[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sigm)
					if h == 60 and w == 60:
						print(j,H1[h+pix[0]*j][w+pix[1]*j])

					total_weight += gaussian(j,sigm)

				H2[h][w] /= total_weight

		print("")

		ind1 = (H2 < 0).astype(int)
		ind2 = ((1 + np.tanh(H2)) < r).astype(int)

		edges = (1 - np.multiply(ind1,ind2))*255

		H2 = np.minimum(H2,edges)

	return edges
