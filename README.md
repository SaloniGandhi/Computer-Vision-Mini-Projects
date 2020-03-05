# Computer_Vision
Computer_Vision Projects - Spring 2020


Project1 - Hybrid Images
Hybrid Images. A hybrid image is the sum of a low-pass filtered version of the one image and a high-pass filtered version of a second image. There is a free parameter, which can be tuned for each image pair, which controls how much high frequency to remove from the first image and how much low frequency to leave in the second image. This is called the "cutoff-frequency". In the paper it is suggested to use two cutoff frequencies (one tuned for each image) and you are free to try that, as well. In the starter code, the cutoff frequency is controlled by changing the standard deviation (sigma) of the Gausian filter used in constructing the hybrid images. 
We provide you with the code for creating a hybrid image, using the functions described above.

This project is intended to familiarize you with Python, NumPy and image filtering. Once you have created an image filtering function, it is relatively straightforward to construct hybrid images.

This project requires you to implement 5 functions each of which builds onto a previous function:

cross_correlation_2d
convolve_2d
gaussian_blur_kernel_2d
low_pass
high_pass

Project2- Feauture Detection and Matching
Synopsis
The goal of feature detection and matching is to identify a pairing between a point in one image and a corresponding point in another image. These correspondences can then be used to stitch multiple images together into a panorama.

In this project, you will write code to detect discriminating features (which are reasonably invariant to translation, rotation, and illumination) in an image and find the best matching features in another image.

To help you visualize the results and debug your program, we provide a user interface that displays detected features and best matches in another image. We also provide an example ORB feature detector, a popular technique in the vision community, for comparison.

