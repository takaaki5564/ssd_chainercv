import copy
import numpy as np

from chainer import reporter
import chainer.training.extensions

from chainercv.evaluations import eval_detection_inria
from chainercv.utils import apply_prediction_to_iterator


class DetectionINRIAEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target, use_07_metric=False, label_names=None):
        super(DetectionINRIAEvaluator, self).__init__(
            iterator, target)
        self.use_07_metric = use_07_metric
        self.label_names = label_names

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterator explicitly
        del imgs

        pred_bboxes, pred_labels, pred_scores = pred_values

        if len(gt_values) == 3:
            gt_bboxes, gt_labels, gt_difficults = gt_values
        elif len(gt_values) == 2:
            gt_bboxes, gt_labels = gt_values
            gt_difficults = None

        result = eval_detection_inria(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=self.use_07_metric)

        report = {'map': result['map']}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
