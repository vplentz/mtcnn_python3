# MTCNN_face_detection_and_alignment using PYTHON3

## About

  This is a python3/mxnet implementation of [Zhang](https://kpzhang93.github.io/)'s work **<Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks>**. it's fast and accurate,  see [link](https://github.com/kpzhang93/MTCNN_face_detection_alignment). 

  It should have **almost** the same output with the original work,  for mxnet fans and those can't afford matlab :)

This is a fork from (https://github.com/pangyupo/mxnet_mtcnn_face_detection) repository.

## Requirement	  

- opencv 

  ​	I use cv2 for image io and resize(much faster than skimage), the input image's channel is acutally BGR

- mxnet 

  ​	If you want to use GPU, you have to install mxnet-cuxx - xx should be replaced by your cuda version (example mxnet-cu92 for Cuda 9.2)

Only tested on Linux in CPU mode

## Test

run:

 ``python3 main.py`` 

you can change `ctx` to `mx.gpu(0)` for faster detection

--- update 20161028 ---

by setting ``num_worker=4``  ``accurate_landmark=False`` we can reduce the detection time by 1/4-1/3, the bboxes are still the same, but we skip the last landmark fine-tune stage( mtcnn_v1 ). 

--- update 20161207 ---

add function `extract_face_chips`, examples:

![1](http://7vikw0.com1.z0.glb.clouddn.com/chip_0.png)
![2](http://7vikw0.com1.z0.glb.clouddn.com/chip_3.png)
![3](http://7vikw0.com1.z0.glb.clouddn.com/chip_2.png)
![4](http://7vikw0.com1.z0.glb.clouddn.com/chip_1.png)

see `mtcnn_detector.py` for the details about the parameters. this function use [dlib](http://dlib.net/)'s align strategy, which works well on profile images :) 
## Results

![big4](http://7xsc78.com1.z0.glb.clouddn.com/face_mtcnn.png)



## License

MIT LICENSE

/

## Reference

K. Zhang and Z. Zhang and Z. Li and Y. Qiao Joint,  Face Detection and Alignment Using Multitask Cascaded Convolutional Networks, IEEE Signal Processing Letters
