import argparse
from argparse import SUPPRESS
import os.path
import main
import main_threaded

parser = argparse.ArgumentParser(description = 'Flow-Based Image Abstraction',usage = SUPPRESS, epilog = "For more details visit https://github.com/ishan00/flow-based-image-abstraction")
parser.add_argument('-f','--file', metavar = 'file_name', required = True, help = 'location of the image to convert')
parser.add_argument('--greyscale', action='store_true', help = 'to specify image is greyscale (default color)')
parser.add_argument('--segmented', action='store_true', help = 'save segmented image')
parser.add_argument('--edge', action='store_true', help = 'save edge image')
parser.add_argument('--batch', action='store_true', help = 'generate multiple images with slightly tweaked parameters')
parser.add_argument('--threading', action='store_true', help = 'Enable Threading')

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
if args.threading:
	main_threaded.main(input_path,output_path,vars(args))
else:
	main.main(input_path,output_path,vars(args))