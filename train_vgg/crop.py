#!/usr/bin/python
# -*- coding: utf-8 -*-
from PIL import Image
import argparse
import os
import numpy as np
import chainer
import sys

train_pos_dir = '../../INRIAPerson/train_64x128_H96/pos'
train_neg_dir = '../../INRIAPerson/train_64x128_H96/neg'
test_pos_dir = '../../INRIAPerson/test_64x128_H96/pos'
test_neg_dir = '../../INRIAPerson/test_64x128_H96/neg'
result_dir = './inria_images'

crop_size = (64, 128)
output_size = (64, 128) # (w, h)


def compute_mean(dataset):
	print('compute mean image')
	
	sum_image = 0
	N = len(dataset)
	for i, (image) in enumerate(dataset):
		sum_image += image
		sys.stderr.write('{} / {}\r'.format(i, N))
		sys.stderr.flush()
	sys.stderr.write('\n')
	
	return sum_image / N


def image_whitening(img):

	img = img.astype(np.float32)
	
	d, w, h = img.shape
	num_pixels = d * w * h
	mean = img.mean()
	variance = np.mean(np.square(img)) - np.square(mean)
	stddev = np.sqrt(variance)
	min_stddev = 1.0 / np.sqrt(num_pixels)
	scale = stddev if stddev > min_stddev else min_stddev
	
	#img -= mean
	img -= mean * 0.1
	#img /= scale
	img /= scale * 9.0
	
	np.clip(img, 0.0, 1.0)	
	return img


def random_contrast(image, lower, upper, seed=None):
	
	f = np.random.uniform(-lower, upper)
	
	mean = (image[0] + image[1] + image[2]).astype(np.float32) / 3
	ximg = np.zeros(image.shape, np.float32)
	for i in range(0, 3):
		ximg[i] = (image[i] - mean) * f + mean
	
	return ximg


def random_brightness(image, max_delta=0.2, seed=None):
	
	delta = np.random.uniform(-max_delta, max_delta)
	newimg = image + delta * 0.2
	
	return newimg


def image_reflect(image):
	b, g, r = image[0, :, :], image[1, :, :], image[2, :, :]
	
	b_new, g_new, r_new = np.fliplr(b), np.fliplr(g), np.fliplr(r)	
	
	newimg = np.zeros(image.shape, np.float32)
	newimg[0, ...] = b_new
	newimg[1, ...] = g_new
	newimg[2, ...] = r_new
	
	return newimg
	

def append_data(imgs, labels, input_dir, label, 
				whitening=False, contrast=False, brightness=False, reflect=False, gap=[0,0]):
	
	cnt = 0
	for source_imgpath in os.listdir(input_dir):
		
		dir_append = ""
		input_img = Image.open(input_dir + "/" + source_imgpath, 'r').convert("RGB")
		width, height = input_img.size
		
		left  = (width - crop_size[0] + gap[0]) / 2
		upper = (height - crop_size[1] + gap[1]) / 2
		right = left + crop_size[0] + gap[0]
		lower = upper + crop_size[1] + gap[1]
		
		if left < 0 or upper < 0 or right > width or lower > height:
			print('illegal crop size: ({0},{1},{2},{3})'.format(left,upper,right,lower))
			continue
		
		crop_img = input_img.crop((left, upper, right, lower))
		
		output_img = crop_img.resize(output_size)
		output_arr = np.asarray(output_img).transpose(2, 0, 1).astype(np.float32) / 255.
		
		if whitening:
			output_arr = image_whitening(output_arr)
			dir_append = dir_append + "w_"
		
		if contrast:
			output_arr = random_contrast(output_arr, lower=0.3, upper=1.5)
			dir_append = dir_append + "c_"
		
		if brightness:
			output_arr = random_brightness(output_arr)
			dir_append = dir_append + "b_"
		
		if reflect:
			output_arr = image_reflect(output_arr)
			dir_append = dir_append + "r_"

		if gap[0] > 0 or gap[1] > 0:
			dir_append = dir_append + "g_"
		
		# append to list object
		imgs.append(output_arr)
		labels.append(label)
		
		cnt = cnt + 1
		
		# save images (for debug)
		#output_img = Image.fromarray(np.uint8(np.asarray(output_arr.transpose(1, 2, 0) * 255)))
		#output_img.save(result_dir + "/" + dir_append + source_imgpath)
		#if cnt >= 5:
		#	break
	
	img_arr = np.asarray(imgs)
	#print('load images N={0} as label={1} from dir={2}'.format(cnt, label, input_dir))
	
	label_arr = np.asarray(labels)


def main():

	print('Start cropping images...')
	train_imgs = []
	train_labels = []
	test_imgs = []
	test_labels = []

	append_data(train_imgs, train_labels, train_pos_dir, 1, False, False, False, False, (0, 0))
	append_data(train_imgs, train_labels, train_neg_dir, 0, False, False, False, False, (0, 0))
	append_data(train_imgs, train_labels, train_pos_dir, 1, True, False, False, False, (0, 0))
	append_data(train_imgs, train_labels, train_neg_dir, 0, True, False, False, False, (0, 0))
	append_data(train_imgs, train_labels, train_pos_dir, 1, False, True, False, False, (0, 0))
	append_data(train_imgs, train_labels, train_neg_dir, 0, False, True, False, False, (0, 0))
	append_data(train_imgs, train_labels, train_pos_dir, 1, False, False, True, False, (0, 0))
	append_data(train_imgs, train_labels, train_neg_dir, 0, False, False, True, False, (0, 0))
	append_data(train_imgs, train_labels, train_pos_dir, 1, False, False, False, True, (0, 0))
	append_data(train_imgs, train_labels, train_neg_dir, 0, False, False, False, True, (0, 0))
	append_data(test_imgs, test_labels, test_pos_dir, 1, False, False, False, False, (0, 0))
	append_data(test_imgs, test_labels, test_pos_dir, 1, True, False, False, False, (0, 0))
	append_data(test_imgs, test_labels, test_pos_dir, 1, False, True, False, False, (0, 0))
	append_data(test_imgs, test_labels, test_pos_dir, 1, False, False, True, False, (0, 0))
	append_data(test_imgs, test_labels, test_pos_dir, 1, False, False, False, True, (0, 0))
	append_data(test_imgs, test_labels, test_neg_dir, 0, False, False, False, False, (0, 0))
	append_data(test_imgs, test_labels, test_neg_dir, 0, True, False, False, False, (0, 0))
	append_data(test_imgs, test_labels, test_neg_dir, 0, False, True, False, False, (0, 0))
	append_data(test_imgs, test_labels, test_neg_dir, 0, False, False, True, False, (0, 0))
	append_data(test_imgs, test_labels, test_neg_dir, 0, False, False, False, True, (0, 0))

	mean = compute_mean(train_imgs)
	np.save('mean.npy', mean)

	np.savez('traindata_vgg', x=train_imgs, y=train_labels)
	np.savez('testdata_vgg', x=test_imgs, y=test_labels)


if __name__ == '__main__':
    main()
