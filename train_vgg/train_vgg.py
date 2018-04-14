#!/usr/bin/env python

from __future__ import print_function

try:
	import matplotlib
	matplotlib.use('Agg')
except ImportError:
	pass

import argparse
import random

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

import chainer.functions as F
from chainer import initializers
import chainer.links as L


class tinyVGG(chainer.Chain):
	
	insize = (96, 48) # (h, w)
	
	def __init__(self):
		super(tinyVGG, self).__init__()
		
		with self.init_scope():
			self.conv1 = L.Convolution2D(None, 32, 5, pad=2)
			self.conv2 = L.Convolution2D(None, 32, 5, pad=2)
			self.conv3 = L.Convolution2D(None, 32, 5, pad=2)
			self.conv4 = L.Convolution2D(None, 32, 5, pad=2)
			self.fc4 = L.Linear(None, 2)
	
	def __call__(self, x): #x:output
		h = F.max_pooling_2d( F.relu(F.dropout( self.conv1(x), ratio=0.5 ) ), 2)
		h = F.relu( F.dropout( self.conv2(h), ratio=0.5 ))
		h = F.max_pooling_2d( F.relu(F.dropout( self.conv3(h), ratio=0.5 ) ), 2)
		h = F.relu( F.dropout( self.conv4(h), ratio=0.5 ))
		h = self.fc4(h)
		
		#h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
		#h = F.relu(self.conv2_1(h))
		#h = F.max_pooling_2d(F.relu(self.conv2_2(h)), 2, stride=2)
		#h = F.relu(self.conv3(h))
		#h = self.fc4(h)
		#loss = F.softmax_cross_entropy(h, t)
		#chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
		#return loss
		
		return h


def load_file( fname ):
	data = np.load(fname)
	
	images = data['x'].astype(np.float32)
	labels = data['y'].astype(np.int32)

	images = images.reshape( images.shape[0], images.shape[1], 96, 48 )
	print('loading file={0} size={1}'.format(fname, images.shape))
	
	return chainer.datasets.TupleDataset(images, labels)


def main():

	parser = argparse.ArgumentParser(description='Training tiny VGG16')
	parser.add_argument('--batchsize', '-B', type=int, default=2,
		                help='Learning minibatch size')
	parser.add_argument('--epoch', '-E', type=int, default=100,
		                help='Number of epochs to train')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
		                help='GPU ID (negative value indicates CPU')
	parser.add_argument('--initmodel',
		                help='Initialize the model from given file')
	parser.add_argument('--resume', '-r', default='',
		                help='Initialize the trainer from given file')
	parser.add_argument('--out', '-o', default='result',
						help='Directory to output the result')
	args = parser.parse_args()
	
	# Initialize the model to train
	tm = tinyVGG()
	model = L.Classifier(tm)
	
	print('Start training tinyVGG')
	
	if args.initmodel:
		print('Load model from', args.initmodel)
		chainer.serializers.load_npz(args.initmodel, model)
	if args.gpu >= 0:
		chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
		model.to_gpu()

	# Load the datasets
	train = load_file("./traindata_vgg.npz")
	test = load_file("./testdata_vgg.npz")
	
	# These iterators load the images with subprocesses running in serial to
	# the training/validation.
	train_iter = chainer.iterators.SerialIterator(train, 100)
	val_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

	# Set up an optimizer
	optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
	#optimizer = chainer.optimizers.Adam(alpha=0.0001)
	optimizer.setup(model)

	# Set up a trainer
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

	#val_interval = (1 if args.test else 100000), 'iteration'
	#log_interval = (1 if args.test else 1000), 'iteration'
	#val_interval = args.epoch, 'epoch'
	val_interval = 10, 'epoch'
	log_interval = 1, 'epoch'

	trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu), trigger=val_interval)
	trainer.extend(extensions.dump_graph('main/loss'))
	trainer.extend(extensions.snapshot(), trigger=val_interval)
	trainer.extend(extensions.snapshot_object(
		model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
	# Reduce the learning rate by half every 25 epochs.
	trainer.extend(
				extensions.ExponentialShift('lr', 0.5, init=1e-4),
				trigger=(25, 'epoch'))

	trainer.extend(extensions.LogReport())
	
	# Save two plot images to the result dir
	if extensions.PlotReport.available():
		trainer.extend(
			extensions.PlotReport(['main/loss', 'validation/main/loss'],
								'epoch', file_name='loss.png'))
		trainer.extend(
			extensions.PlotReport(
				['main/accuracy', 'validation/main/accuracy'],
				'epoch', file_name='accuracy.png'))
	
	trainer.extend(extensions.observe_lr(), trigger=log_interval)
	trainer.extend(extensions.PrintReport([
		'epoch', 'main/loss', 'main/accuracy', 
		'validation/main/accuracy', 'validation/main/loss',
	]), trigger=log_interval)
	trainer.extend(extensions.ProgressBar(update_interval=1))

	if args.resume:
		chainer.serializers.load_npz(args.resume, trainer)

	trainer.run()
	
	model.to_cpu()
	chainer.serializers.save_npz("tinyvgg_model.npz", model)
	print('save model completed.')


if __name__ == '__main__':
    main()