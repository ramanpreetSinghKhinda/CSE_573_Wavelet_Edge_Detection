<p align="center">Computer Vision and Image Processing</br>Wavelet Edge Detection</br>CSE 573 - Fall 2015
==========================================================================================

<p align="center">![Img_1](https://raw.githubusercontent.com/ramanpreet1990/CSE_573_Wavelet_Edge_Detection/master/Resources/1.png)


Goal
------
In this project, we have to design a wavelet transform based scheme for edge detection. In particular, the wavelet transforms at two adjacent scales are multiplied in order to magnify edge structure and suppress noise. The following steps are to be performed:

(a) Select from several possible wavelet transforms and determine which two to use with convincing reasons

(b) Add Gaussian noise and impulse noise, respectively, to the test images for your implementation; You may select the level of noise with explanation in your report

(c) Perform the selected wavelet transforms to at least three different test images and their noise corrupted versions; The transform should be performed for at least four levels

(d) Perform proper scale multiplications to each of the wavelet transformed images and combine the results at different levels to form the final edge map

(e) Display the test images and their final edge detection results based on two different wavelet transforms.

(f) Compare the results of edge detection from two different wavelet transforms and from two different noise types; Explain the difference among these results

(g) Bonus (5%): Implement an edge detection scheme combining wavelet


Literature Review
----
Edge detection is a very prevalent problem in the field of image processing and has been researched a lot. Some edge detection techniques fall in the spatial domain edge detection like Canny and sobel. The other category is edge detection in the frequency domain. 

Mallat and Zhong in their work[1] proposed a dyadic wavelet and a corresponding wavelet transform. The idea was to detect edges in the wavelet domain by applying wavelet transform to the image in different scales. It was observed that the edge structures were visible in each subband but the noise levels decreased rapidly along these scales.

Xu et al[2] presented a method in which they enhanced significant structures by multiplying the adjacent DWT levels and Sadler and Swami[3] analyzed the multiscale product in step detection and estimation . The paper[4] we are following does something along these lines; we make a multiscale edge detector by multiplying adjacent subbands and making edge maps at different scales and combining them in the end. Edges are determined as local maxima in the product function after thresholding.

The pipeline majorly includes four image processing techniques: convolution of image with a filter as the wavelets used are essentially filters to be convoluted with the images, downsampling the image to half its size by removing alternate rows and columns from the image, element wise multiplication of image arrays and up-sampling the images to twice their sizes by using some kind of interpolation techniques. 


Credits
-------
We acknowledge and grateful to [**Professor Chang Wen Chen**](http://www.cse.buffalo.edu/faculty/chencw/) and TAs [**Radhakrishna Dasari**](http://www.acsu.buffalo.edu/~radhakri/) and [**Bhargava Urala Kota**](http://www.cse.buffalo.edu/people/?u=buralako) for their continuous support throughout the Course ([**CSE 573**](http://www.cse.buffalo.edu/shared/course.php?e=CSE&n=573&t=Comp+Vision+%26+Image+Proc)) that helped us learn the skills of Computer Vision and design a wavelet transform based scheme for edge detection.


Contributors
---------
[Apoorva Mittal](https://www.linkedin.com/in/apoorva-mittal-0b524357)


Ramanpreet Singh Khinda (rkhinda@buffalo.edu)</br>
[![website](https://raw.githubusercontent.com/ramanpreet1990/CSE_586_Simplified_Amazon_Dynamo/master/Resources/ic_website.png)](https://branded.me/ramanpreet1990)		[![googleplay](https://raw.githubusercontent.com/ramanpreet1990/CSE_586_Simplified_Amazon_Dynamo/master/Resources/ic_google_play.png)](https://play.google.com/store/apps/details?id=suny.buffalo.mis.research&hl=en)		[![twitter](https://raw.githubusercontent.com/ramanpreet1990/CSE_586_Simplified_Amazon_Dynamo/master/Resources/ic_twitter.png)](https://twitter.com/dk_sunny1)		[![linkedin](https://raw.githubusercontent.com/ramanpreet1990/CSE_586_Simplified_Amazon_Dynamo/master/Resources/ic_linkedin.png)](https://www.linkedin.com/in/ramanpreet1990)


License
----------
Copyright {2016} 
{Ramanpreet Singh Khinda rkhinda@buffalo.edu and Apoorva Mittal amittal2@buffalo.edu} 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
