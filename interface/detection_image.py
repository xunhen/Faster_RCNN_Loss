import numpy as np
import tensorflow as tf
from PIL import Image

from distutils.version import StrictVersion
from Lib import model_builder
from object_detection.utils import config_util
from Tool.generate_box_vibe import Generate_Box_By_ViBe
import cv2

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


class Detection(object):
    def __init__(self, pipeline_config_path, restore_path, filter_threshold=0.5):
        self._rpn_type = None
        self._filter_fn_arg = {'filter_threshold': filter_threshold}
        self._get_filter_boxes_fn = Generate_Box_By_ViBe()
        self._pipeline_config_path = pipeline_config_path
        self._restore_path = restore_path
        self._replace_rpn_arg = None
        self._graph = None
        pass

    def init_some(self):
        pass

    def _process(self):
        pass

    def test(self):
        print(self.__pipeline_config_path)
        print(self._restore_path)

    def build_model(self):
        print('build_model_begin')
        batch_size = 1  # detect one image once
        configs = config_util.get_configs_from_pipeline_file(self._pipeline_config_path)
        model_config = configs['model']
        self._graph = tf.Graph()
        with self._graph.as_default():
            model = model_builder.build(model_config=model_config, is_training=False, rpn_type=self._rpn_type,
                                        filter_fn_arg=self._filter_fn_arg, replace_rpn_arg=self._replace_rpn_arg)

            with tf.variable_scope('placeholder'):
                self._image_tensor = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3), name='images')
                if self._filter_fn_arg:
                    self._filter_box_list = [tf.placeholder(tf.float32, shape=(None, 4), name='filter_box{}'.format(i))
                                             for i in range(batch_size)]
                    model.provide_filter_box_list(self._filter_box_list)

            self._preprocessed_inputs, self._true_image_shapes = model.preprocess(self._image_tensor)
            self._prediction_dict = model.predict(self._preprocessed_inputs, self._true_image_shapes)
            self._output_dict = model.postprocess(self._prediction_dict, self._true_image_shapes)

            self._saver = tf.train.Saver()
            self._sess = tf.Session()
            self._saver = tf.train.Saver()
            self._saver.restore(sess=self._sess, save_path=self._restore_path)
        print('build_model_end')

    def detection(self, image, gray_image=None):
        # when using filter ,the gray_image is necessary!!
        print('detection_begin')
        image_expanded = np.expand_dims(image, axis=0)
        feed_dict = {self._image_tensor: image_expanded}
        if self._filter_fn_arg:
            if gray_image is None:
                print('when using filter ,the gray_image is necessary!!')
            bboxes = self._get_filter_boxes_fn.processAndgenerate(gray_image)
            feed_dict[self._filter_box_list[0]] = bboxes  # only batch_size=1
        output_dict_, prediction_dict_ = self._sess.run([self._output_dict, self._prediction_dict],
                                                        feed_dict=feed_dict)
        output_dict_['detection_classes'][0] += 1
        result = np.concatenate(
            (output_dict_['detection_boxes'][0], np.expand_dims(output_dict_['detection_classes'][0], axis=1)), axis=1)
        print(result)
        print('detection_end')
        # return [output_dict_['detection_boxes'][0], output_dict_['detection_classes'][0]]
        return [np.array(result, dtype='double'), ]


if __name__ == '__main__':
    pipeline_config_path = r'..\Model\pipeline\pipeline_resnet50.config'
    restore_path = r'..\log\train_org\model.ckpt-200000'
    image_path = r'F:\\PostGraduate\\Projects\\background\\video\\post\\1.jpg'
    detection = Detection(pipeline_config_path, restore_path)
    detection.build_model()
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    image = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(detection.detection(image, image_gray))
    print(detection.detection(image, image_gray))
    pass
