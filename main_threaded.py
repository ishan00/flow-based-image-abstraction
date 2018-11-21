import cv2
import numpy as np
import ETF
import FDoG_threaded
import FBL_threaded
from utils import *
import os
import pickle
import threading
import time
# from scipy.io import savemat

def main(input_path, output_path, flags):

	start = time.time()
	name,ext = os.path.splitext(input_path)
	name = name.split('/')[-1]

	image = cv2.imread(input_path, 1)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	if not os.path.isdir('tmp'):
		os.mkdir('tmp')

	if os.path.isfile('tmp/' + name + '_etf.pickle'):
		with open('tmp/' + name + '_etf.pickle', 'rb') as file:
			etf = pickle.load(file)
			
			print_progress('ETF Horiz',0,1)
			print("")
			print_progress('ETF Vert',0,1)
			print("")

	else:
		
		etf = ETF.ETF(gray_image, iterations = 1, mu = 5, type = 0)
		
		with open('tmp/' + name + '_etf.pickle', 'wb') as file:
			pickle.dump(etf,file)

	t1_created = False
	if os.path.isfile('tmp/' + name + '_edges.pickle'):
		with open('tmp/' + name + '_edges.pickle', 'rb') as file:
			edges = pickle.load(file)

			print_progress('DOG',0,1)
			print("")
			print_progress('Gaussian',0,1)
			print("")

	else:
		t1_created = True
		t1 = threading.Thread(target = FDoG_threaded.FDoG, args = (gray_image, etf, 3, True,))
		t1.start()

	t2_created = False
	if os.path.isfile('tmp/' + name + '_smooth.pickle'):
		with open('tmp/' + name + '_smooth.pickle', 'rb') as file:
			smoothed_image = pickle.load(file)

			print_progress('FBL S',0,1)
			print("")
			print_progress('FBL T',0,1)
			print("")

	else:
		t2_created = True
		t2 = threading.Thread(target = FBL_threaded.FBL, args = (image, etf, 1,))
		t2.start()
	
	if t1_created:
		
		t1.join()
		with open('tmp/' + name + '_edges.pickle', 'wb') as file:
			pickle.dump(FDoG_threaded.edges,file)

		edges = FDoG_threaded.edges
	
	if t2_created:

		t2.join()
		with open('tmp/' + name + '_smooth.pickle', 'wb') as file:
			pickle.dump(FBL_threaded.smoothed_image,file)

		smoothed_image = FBL_threaded.smoothed_image


	size = image.shape

	images = color_segmentation(smoothed_image.astype(np.uint8), batch = False)

	for j in range(len(edges)):
		name,ext = os.path.splitext(input_path)
		output_path = os.path.join(name + '_out_'  + str(j) + ext)
		
		var = np.minimum(images,np.stack([edges[j], edges[j], edges[j]], axis=2))	
		cv2.imwrite(output_path,var)

	end = time.time()

	print ('Total - ',end - start)