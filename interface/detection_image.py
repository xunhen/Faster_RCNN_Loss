import numpy as np
import tensorflow as tf

from distutils.version import StrictVersion
from Lib import model_builder
from object_detection.utils import config_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


class Detection(object):
    def __init__(self, pipeline_config_path, restore_path):
        self._rpn_type = None
        self._replce_rpn_arg = None
        self._filter_fn_arg = None
        self._graph = None
        self._pipeline_config_path = pipeline_config_path
        self._restore_path = restore_path
        pass

    def init_some(self):
        pass

    def _process(self):
        pass

    def test(self):
        print(self.__pipeline_config_path)
        print(self._restore_path)

    def _build_model(self):
        batch_size = 1  # detect one image once
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)
        model_config = configs['model']
        self._graph = tf.Graph()
        with self._graph.as_default():
            model = model_builder.build(model_config=model_config, is_training=False, rpn_type=self._rpn_type,
                                        filter_fn_arg=self._filter_fn_arg, replace_rpn_arg=self._replace_rpn_arg)

            with tf.variable_scope('placeholder'):
                self._image_tensor = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3), name='images')
            self._preprocessed_inputs, self._true_image_shapes = model.preprocess(self._image_tensor)
            self._prediction_dict = model.predict(self._preprocessed_inputs, self._true_image_shapes)
            self._output_dict = model.postprocess(self._prediction_dict, self._true_image_shapes)

            self._saver = tf.train.Saver()

            self._sess = tf.Session()
            self._saver = tf.train.Saver()
            self._saver.restore(sess=self._sess, save_path=self._restore_path)

        pass

    def detection(self, image):
        image_expanded = np.expand_dims(image, axis=0)
        feed_dict = {self._image_tensor: image_expanded}
        output_dict_, prediction_dict_ = self._sess.run([self._output_dict, self._prediction_dict],
                                                        feed_dict=feed_dict)
        output_dict_['detection_classes'][0] += 1
        return output_dict_['detection_boxes'][0]


if __name__ == '__main':
    pass
