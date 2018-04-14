import os

from chainer.dataset import download

from chainercv import utils

def get_inria(split):
    base_path = os.path.join('../../INRIAPerson')
    split_file = os.path.join('../../INRIAPerson/' + split + '/pos.lst')
    
    return base_path


inria_bbox_label_names = (
#    'bicycle',
#    'bus',
#    'car',
#    'motorbike',
    'person',
#    'train',
    )

