
import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import sobel
from scipy.ndimage import prewitt
from scipy.ndimage import gaussian_filter1d


IMG_FILE = 'tutorials/image_filtering/data/mri_prostate.dat'

dtype_in_use = 'float32'

img0 = np.genfromtxt(IMG_FILE,delimiter=',',dtype='uint8')
Image.fromarray(img0).save('tutorials/image_filtering/data/filtering_output/original.png')


## smoothing
kernel = np.ones((3,3),dtype=dtype_in_use)
img1 = convolve2d(img0.astype(dtype_in_use),kernel/kernel.size,mode='same')
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/ones_3by3.png')

kernel = np.ones((7,7),dtype=dtype_in_use)
img1 = convolve2d(img0.astype(dtype_in_use),kernel/kernel.size,mode='same')
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/ones_7by7.png')

kernel = np.ones((9,9),dtype=dtype_in_use)
img1 = convolve2d(img0.astype(dtype_in_use),kernel/kernel.size,mode='same')
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/ones_9by9.png')

kernel = np.ones((3,3),dtype=dtype_in_use)
img1 = convolve2d(img0.astype(dtype_in_use),kernel/kernel.size,mode='same')
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/ones_3by3.png')

img1 = gaussian_filter(img0.astype(dtype_in_use),sigma=3)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/gaussian_sigma3.png')

img1 = gaussian_filter(img0.astype(dtype_in_use),sigma=15)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/gaussian_sigma15.png')

img1 = median_filter(img0.astype(dtype_in_use),size=(3,3))
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/median_3by3.png')

img1 = median_filter(img0.astype(dtype_in_use),size=(7,7))
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/median_7by7.png')


## Differentiation
img1 = sobel(img0.astype(dtype_in_use),axis=0)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/sobel_d0.png')
img1 = sobel(img0.astype(dtype_in_use),axis=1)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/sobel_d1.png')

img1 = prewitt(img0.astype(dtype_in_use),axis=0)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/prewitt_d0.png')
img1 = prewitt(img0.astype(dtype_in_use),axis=1)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/prewitt_d1.png')

img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=1,order=1,axis=0)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/derivative_s1_d0.png')
img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=1,order=1,axis=1)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/derivative_s1_d1.png')

img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=3,order=1,axis=0)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/derivative_s3_d0.png')
img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=3,order=1,axis=1)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/derivative_s3_d1.png')

img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=1,order=2,axis=0)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/laplacian_s1_d0.png')
img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=1,order=2,axis=1)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/laplacian_s1_d1.png')

img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=3,order=2,axis=0)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/laplacian_s3_d0.png')
img1 = gaussian_filter1d(img0.astype(dtype_in_use),sigma=3,order=2,axis=1)
Image.fromarray(img1.astype('uint8')).save('tutorials/image_filtering/data/filtering_output/laplacian_s3_d1.png')
