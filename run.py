import argparse
from argparse import SUPPRESS
import os.path
from main import main

parser = argparse.ArgumentParser(description = 'Flow-Based Image Abstraction',usage = SUPPRESS, epilog = "For more details visit https://github.com/ishan00/flow-based-image-abstraction")
parser.add_argument('-f','--file', metavar = 'file_name', required = True, help = 'location of the image to convert')
parser.add_argument('--greyscale', action='store_true', help = 'to specify image is greyscale (default color)')
parser.add_argument('--segmented', action='store_true', help = 'save segmented image')
parser.add_argument('--edge', action='store_true', help = 'save edge image')
parser.add_argument('--multi', action='store_true', help = 'generate multiple images with slightly tweaked parameters')

args = parser.parse_args()

print (args)

input_path = args.file

if not os.path.isfile(input_path):
	parser.error("File " + str(input_path) + " does not exist")

name,ext = os.path.splitext(input_path)
output_path = os.path.join(name + '_out' + ext)

flags = vars(args)
del flags['file']

#print (flags)

main(input_path,output_path,vars(args))