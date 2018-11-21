from utils import *
import numpy as np
import threading

lock = threading.Lock()

edges = []
modified_image = []

def FDoG_iter(image, etf):

	size = image.shape

	H2 = image
	H1 = np.zeros(size)

	for h in range(size[0]):
		
		print_progress('DOG',h,size[0])

		for w in range(size[1]):

			total_weight = 0
			angle_perpendicular = (angle(etf[h][w][0],etf[h][w][1]) + np.pi/2)
			
			if angle_perpendicular > 2*np.pi:
				angle_perpendicular -= 2*np.pi

			pix = flow_neighbour(angle_perpendicular*180/np.pi)

			for j in range(-3,4):

				if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue 

				H1[h][w] += H2[h+pix[0]*j][w+pix[1]*j]*dog_filter(abs(j),sigc,p)


	print("")

	H2 = np.zeros(size)

	for h in range(size[0]):

		print_progress('Gaussian',h,size[0])
		
		for w in range(size[1]):

			total_weight = 0
			ang = angle(etf[h][w][0],etf[h][w][1])
			pix = flow_neighbour(ang*180/np.pi)

			for j in range(-3,4):

				if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue

				H2[h][w] += H1[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sigm)
				

				total_weight += gaussian(j,sigm)

			H2[h][w] /= total_weight

	print("")

	return H2

def FDoG_thread(i, etf, iter):

	lock.acquire()
	image = modified_image[i]
	lock.release()

	size = image.shape
	
	H2 = image

	for _ in range(iter):
		
		H1 = np.zeros(size)

		for h in range(size[0]):
			print_progress('DOG ' + str(i) ,h,size[0])
			for w in range(size[1]):

				total_weight = 0
				angle_perpendicular = (angle(etf[h][w][0],etf[h][w][1]) + np.pi/2)
				
				if angle_perpendicular > 2*np.pi:
					angle_perpendicular -= 2*np.pi

				pix = flow_neighbour(angle_perpendicular*180/np.pi)

				for j in range(-3,4):

					if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue 

					H1[h][w] += H2[h+pix[0]*j][w+pix[1]*j]*dog_filter(abs(j),sigc,p)

		print("")

		H2 = np.zeros(size)

		for h in range(size[0]):

			print_progress('Gaussian ' + str(i),h,size[0])
			
			for w in range(size[1]):

				total_weight = 0
				ang = angle(etf[h][w][0],etf[h][w][1])
				pix = flow_neighbour(ang*180/np.pi)

				for j in range(-3,4):

					if h + pix[0]*j < 0 or h + pix[0]*j >= size[0] or w + pix[1]*j < 0 or w + pix[1]*j >= size[1]: continue

					H2[h][w] += H1[h+pix[0]*j][w+pix[1]*j]*gaussian(abs(j),sigm)
					

					total_weight += gaussian(j,sigm)

				H2[h][w] /= total_weight

		print("")

		ind1 = (H2 < 0).astype(int)
		ind2 = ((1 + np.tanh(H2)) < R[i]).astype(int)

		edge = (1 - np.multiply(ind1,ind2))*255

		H2 = np.minimum(image,edge)

	lock.acquire()
	edges[i] = edge
	lock.release()

def FDoG(image, etf, iterations = 1, batch = False):

	global edges,modified_image

	H2 = FDoG_iter(image,etf)

	if batch:

		ind1 = (H2 < 0).astype(int)

		edges = [0 for _ in range(len(R))]
		modified_image = [0 for _ in range(len(R))]

		for j in range(len(R)):
			ind2 = ((1 + np.tanh(H2)) < R[j]).astype(int)
			edges[j] = (1 - np.multiply(ind1,ind2))*255
			modified_image[j] = np.minimum(image, edges[j])

		threads_list = []

		for j in range(len(R)):
			t1 = threading.Thread(target = FDoG_thread, args = (j,etf,iterations-1))
			threads_list.append(t1)
			t1.start()

		for j in range(len(R)):
			threads_list[j].join()

		return edges

	else:

		ind1 = (H2 < 0).astype(int)
		ind2 = ((1 + np.tanh(H2)) < r).astype(int)

		edges = (1 - np.multiply(ind1,ind2))*255
		modified_image = np.minimum(image,edges)

		for i in range(iterations-1):

			edges = FDoG_iter(modified_image, etf)

			ind1 = (edges < 0).astype(int)
			ind2 = ((1 + np.tanh(edges)) < r).astype(int)

			edges = (1 - np.multiply(ind1,ind2))*255
			modified_image = np.minimum(modified_image,edges)

		return edges






