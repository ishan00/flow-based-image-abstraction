import cv2
import numpy as np
from ETF import ETF
from FDoG import FDoG, FDoG_iter
from FBL import FBL
from utils import *
import os
import pickle
import time

def main(input_path, output_path, flags):

	batch = flags['batch']
	greyscale = flags['greyscale']

	print (greyscale)

	start = time.time()

	name,ext = os.path.splitext(input_path)
	
	name = os.path.basename(name)
	name = os.path.splitext(name)[0]
	#name = name.split('/')[-1]

	image = cv2.imread(input_path, 1)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	if not os.path.isdir('tmp'):
		os.mkdir('tmp')

	if os.path.isfile(os.path.join('tmp',name+'_etf.pickle')):
		with open(os.path.join('tmp',name+'_etf.pickle'), 'rb') as file:
			etf = pickle.load(file)
			
			print_progress('ETF Horiz',0,1)
			print("")
			print_progress('ETF Vert',0,1)
			print("")

	else:
		
		time1 = time.time()
		etf = ETF(gray_image, iterations = 1, mu = 5, type = 0)
		time2 = time.time()

		print ('ETF - ', time2 - time1)

		with open(os.path.join('tmp',name+'_etf.pickle'), 'wb') as file:
			pickle.dump(etf,file)


	if os.path.isfile(os.path.join('tmp',name+'_edges.pickle')):
		with open(os.path.join('tmp',name+'_edges.pickle'), 'rb') as file:
			edges = pickle.load(file)

			print_progress('DOG',0,1)
			print("")
			print_progress('Gaussian',0,1)
			print("")

	else:
		
		time1 = time.time()
		edges = FDoG(gray_image, etf, iterations = 1, batch = True)
		time2 = time.time()

		print ('FDoG - ', time2 - time1)

		with open(os.path.join('tmp',name+'_edges.pickle'), 'wb') as file:
			pickle.dump(edges,file)


	if os.path.isfile(os.path.join('tmp',name+'_smooth.pickle')):
		with open(os.path.join('tmp',name+'_smooth.pickle'), 'rb') as file:
			smoothed_image = pickle.load(file)

			print_progress('FBL S',0,1)
			print("")
			print_progress('FBL T',0,1)
			print("")

	else:

		time1 = time.time()
		smoothed_image = FBL(image, etf, iterations = 1)
		time2 = time.time()

		print ('FBL - ',time2 - time1)

		with open(os.path.join('tmp',name+'_smooth.pickle'), 'wb') as file:
			pickle.dump(smoothed_image,file)

	size = image.shape

	images = color_segmentation(smoothed_image.astype(np.uint8), batch, greyscale)

	for j in range(len(edges)):
		name,ext = os.path.splitext(input_path)
		output_path = os.path.join(name + '_out_'  + str(j) + ext)
		
		var = np.minimum(images,np.stack([edges[j], edges[j], edges[j]], axis=2))	
		cv2.imwrite(output_path,var)

	end = time.time()

	print ('Total - ', end - start)
