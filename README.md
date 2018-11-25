# flow-based-image-abstraction
This repository is the implementation of the paper [Flow-Based Image Abstraction](http://www.cs.umsl.edu/~kang/Papers/kang_tvcg09.pdf) by Henry Kang, Seungyong Lee and Charles K. Chui.

### Running
```python
python3 run.py --file FILE_PATH
```
By default the program will generate 5 images with different amount of edges. The following flags can be used. 

`--threading` to enable threading. This will create separate threads for FBL and FDoG. Also FDoG for different values of r will also be calculated in separate threads. Though we didn't see much speedup by enabling threading.

`--greyscale` flag will remove chrominance from the image resulting in greyscale image.

The program takes around 5 min for 600*800 images. If you have a very large image, please reduce the size.

Other flags will be added later

### Outputs
Here are few of the samples generated so far. The leftmost is the original image, the middle one and the right figures are generated using r = 0.5 and 0.8 for FDoG.

![](images/merged_buildings.jpg)

![](images/merged_construction.jpg)

![](images/merged_dogs.jpg)

![](images/merged_sydney.jpg)

### More outputs
The left image is the original whereas the right image is the best looking among r = 0.5,0.6,0.7,0.8

![](images/merged_forest.jpg)

![](images/merged_snow.jpg)

![](images/merged_street.jpg)

### Still more outputs

The following were created by running the above program on each frame of video and then merging them.

<img style="width:100%" src="images/mrbean.gif">

<img style="width:100%" src="images/road.gif">

