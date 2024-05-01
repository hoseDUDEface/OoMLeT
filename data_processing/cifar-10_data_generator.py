import os.path
from typing import List, Dict, Any, Optional

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from Common.miscellaneous import print_indexed_list
from data_processing.input_constants import KEYS_TO_FEATURES, LANDMARK_KEYS
from data_processing import tf_augmentations
from data_processing.solo_records.single_heatmap_encoding import generate_bboxes_heatmap_tf, generate_landmarks_heatmaps_tf


class DataGeneratorXY(object):
    def __init__(self, X_data, Y_data, indice=[], desired_image_size: Optional[int] = 128, batch_size: int = 4,
                 epochs: Optional[int] = None, augmentations: bool = False, data_format: str = 'channels_first',
                 stop_repeat: bool = False, training_labels_only: bool = False, batch_balancing_data_ratios_dict: Optional[Dict] = None,
                 shuffle: bool = True, scale_values: bool = False, heatmap_scale: int = 4, bbox_sigma_divisor: float = 3., lms_sigma_divisor: float = 10.,
                 training: bool = False):
        self.X_data = np.array(X_data)
        self.Y_data = np.array(Y_data)
        self.indice = indice

        self.desired_image_size = desired_image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.augmentations = augmentations
        self.data_format = data_format
        self.stop_repeat = stop_repeat
        self.training_labels_only = training_labels_only
        self.batch_balancing_data_ratios_dict = batch_balancing_data_ratios_dict
        self.shuffle = shuffle
        self.scale_values = scale_values
        self.heatmap_scale = heatmap_scale
        self.bbox_sigma_divisor = bbox_sigma_divisor
        self.lms_sigma_divisor = lms_sigma_divisor
        self.training = training

        self.output_size = self.desired_image_size // self.heatmap_scale

        if self.shuffle:
            np.random.shuffle(self.tf_record_filenames)

        if training:
            self.make_dataset_endoder()

        # self.global_step = 0
        self.reset()

    def reset(self):
        if self.batch_balancing_data_ratios_dict:
            separated_datasets, dataset_weights = self.separate_datasets_over_data()
            self.dataset = tf.data.experimental.sample_from_datasets(separated_datasets, dataset_weights)
        else:
            self.dataset = tf.data.TFRecordDataset(self.tf_record_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)

        self.dataset = self.dataset.map(map_func=self.parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)

        if not self.stop_repeat:
            self.dataset = self.dataset.repeat()

        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def parse_fn(self, serialized_example):
        parsed_example = tf.io.parse_single_example(serialized_example, KEYS_TO_FEATURES)

        inputs_dict = self.process_inputs(parsed_example)
        labels_dict = self.process_labels(parsed_example)

        return inputs_dict, labels_dict

    def process_inputs(self, parsed_example):
        image = tf.io.decode_jpeg(parsed_example['image/encoded'], channels=3)

        aug_type = None
        if self.augmentations:
            image, aug_type = self.augment(image)

        image = tf.cast(image, tf.float32)
        image = tf_augmentations.resize_and_pad_tf_image(image, self.desired_image_size)

        preprocessed_image = self.preprocess_image(image)

        return preprocessed_image

    def process_labels(self, parsed_example):
        # parsed_example = set_default_class(parsed_example, 'labels/face_classes', -1, 1)
        # parsed_example = set_default_class(parsed_example, 'labels/mask_classes', -1, 0)

        labels_dict = {}
        metadata_dict = {}
        augmentation_dict = {}
        weights_dict = {}

        for key, value in parsed_example.items():
            stripped_key = key.split('/')[-1]

            if key.startswith('metadata'):
                metadata_dict[stripped_key] = value

            elif key.startswith('augmentation'):
                augmentation_dict[stripped_key] = value

            elif key.startswith('weights'):
                weights_dict[stripped_key] = value

            elif 'landmarks' in key:  # landmarks are processed separately
                labels_dict[stripped_key] = value
                continue

            elif key.startswith('labels'):
                labels_dict[stripped_key] = value

        # print("parsed_example.keys()", parsed_example.keys())

        # landmark_list = [tf.sparse.to_dense(parsed_example[landmark_key], default_value=-1) for landmark_key in LANDMARK_KEYS]
        # labels_dict['landmarks'] = tf.stack(landmark_list, axis=0)

        labels_dict['metadata'] = metadata_dict
        labels_dict['augmentation'] = augmentation_dict
        labels_dict['weights'] = weights_dict
        # record_name = metadata_dict['record_name']

        # dense_xmins = tf.sparse.to_dense(labels_dict['x_mins'], default_value=-1)
        # dense_ymins = tf.sparse.to_dense(labels_dict['y_mins'], default_value=-1)
        # dense_xmaxs = tf.sparse.to_dense(labels_dict['x_maxs'], default_value=-1)
        # dense_ymaxs = tf.sparse.to_dense(labels_dict['y_maxs'], default_value=-1)
        # bboxes = tf.stack([dense_xmins, dense_ymins, dense_xmaxs, dense_ymaxs], axis=1)
        #
        # bboxes_heatmap, radii = generate_bboxes_heatmap_tf(bboxes, (self.output_size, self.output_size), self.bbox_sigma_divisor)
        #
        # labels_dict['bboxes_heatmaps'] = bboxes_heatmap
        #
        # landmarks_list = [tf.sparse.to_dense(parsed_example[landmark_key], default_value=-1) for landmark_key in LANDMARK_KEYS]
        # landmarks = tf.stack([tf.reshape(landmark, (-1, 2)) for landmark in landmarks_list], axis=1)  # (landmark.shape[-1] // 2, 2)
        #
        # landmarks_heatmaps = generate_landmarks_heatmaps_tf(landmarks, radii, (self.output_size, self.output_size), self.lms_sigma_divisor)
        #
        # labels_dict['landmarks_heatmaps'] = landmarks_heatmaps

        return labels_dict

    def generate(self):
        for record in self.dataset:
            yield record
        print("Done generating dataset")

    def preprocess_image(self, resized_inputs: tf.Tensor) -> tf.Tensor:
        """
        Preprocesses the image for the model.
        :param resized_inputs: Images of the right dimensions.
        :return: Preprocessed images.
        """
        preprocessed_inputs = (resized_inputs - 127.5) / 127.5

        if self.data_format == 'channels_first':
            if len(preprocessed_inputs.shape) > 3:
                preprocessed_inputs = tf.transpose(preprocessed_inputs, [0, 3, 1, 2])
            else:
                preprocessed_inputs = tf.transpose(preprocessed_inputs, [2, 0, 1])

        return preprocessed_inputs

    def unprocess_image(self, preprocessed_images):
        unprocessed_inputs = preprocessed_images * 127.5 + 127.5

        if self.data_format == 'channels_first':
            if len(unprocessed_inputs.shape) > 3:
                unprocessed_inputs = tf.transpose(unprocessed_inputs, [0, 2, 3, 1])
            else:
                unprocessed_inputs = tf.transpose(unprocessed_inputs, [1, 2, 0])

        return unprocessed_inputs

    def make_dataset_endoder(self):
        # record_path_len = len(self.tf_record_filenames[0].split('/'))
        # dataset_name_path_index =
        dataset_names = [record_name.split('/')[-3] for record_name in self.tf_record_filenames]
        unique_dataset_names = sorted(list(set(dataset_names)))
        print("unique_dataset_names", unique_dataset_names)

        # set_names = [record_name.split('/')[-2] for record_name in self.tf_record_filenames]
        # unique_set_names = list(set(set_names))
        # print("unique_set_names", unique_set_names)

        self.dataset_name_to_int = {dataset_name: i for i, dataset_name in enumerate(unique_dataset_names)}
        self.int_to_dataset_name = {i: dataset_name for dataset_name, i in self.dataset_name_to_int.items()}

        self.set_to_int = {'train': 0, 'val': 1, 'test': 2}
        self.int_to_set = {i: set_name for set_name, i in self.set_to_int.items()}

        print("self.dataset_name_to_int", self.dataset_name_to_int)
        # table = tf.lookup.StaticHashTable(
        #     tf.lookup.KeyValueTensorInitializer(self.dataset_name_to_int.keys(), self.dataset_name_to_int.values(), key_dtype=tf.int64, value_dtype=tf.int64), -1
        # )
        self.dataset_name_to_int_tf = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.convert_to_tensor(list(self.dataset_name_to_int.keys()), tf.string), tf.convert_to_tensor(list(self.dataset_name_to_int.values()), tf.int64),
                key_dtype=tf.string, value_dtype=tf.int64,
            ), num_oov_buckets=1)

        self.set_to_int_tf = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(self.set_to_int.keys()), list(self.set_to_int.values()),
                key_dtype=tf.string, value_dtype=tf.int64,
            ), num_oov_buckets=1)

    def encode_record_name(self, record_name):
        dataset_name, set_name, record_index = record_name.split('/')

        record_true_name, record_str = os.path.splitext(record_index)
        face_id, aug_index = record_true_name.split('-')
        face_id = int(face_id)
        aug_index = int(aug_index)

        dataset_name_int = self.dataset_name_to_int[dataset_name]
        set_name_int = self.set_to_int[set_name]

        return dataset_name_int, set_name_int, face_id, aug_index

    def encode_record_name_tf(self, record_name):
        # dataset_name, set_name, record_index = record_name.split('/')
        record_name_split = tf.strings.split(record_name, '/')
        record_true_name = tf.strings.split(record_name_split[2], '.')[0]
        record_true_name_split = tf.strings.split(record_true_name, '-')

        face_id = tf.strings.to_number(record_true_name_split[0], tf.int64)
        aug_index = tf.strings.to_number(record_true_name_split[1], tf.int64)

        dataset_name_int = self.dataset_name_to_int_tf[record_name_split[0]]
        set_name_int = self.set_to_int_tf[record_name_split[1]]

        return dataset_name_int, set_name_int, face_id, aug_index

    def decode_record_name(self, dataset_name_int, set_name_int, face_id, aug_index):
        dataset_name = self.int_to_dataset_name[dataset_name_int]
        set_name = self.int_to_set[set_name_int]

        record_index = "{:05d}-{}.record".format(int(face_id), int(aug_index))

        record_name = os.path.join(dataset_name, set_name, record_index)

        return record_name

    def separate_datasets_over_data(self):
        assert sum(self.batch_balancing_data_ratios_dict.values()) - 1 < 0.001, "The sum of data type weights does not sum to 1. " \
                                                                         "Sum = {}".format(sum(self.batch_balancing_data_ratios_dict.values()))
        # separated_record_files = []
        dataset_weights = []
        categorized_datasets = []
        total_collected_samples = 0

        for data_type, balancing_ratio in self.batch_balancing_data_ratios_dict.items():
            dataset_records = [record_name for record_name in self.tf_record_filenames if record_name.split('/')[-3] == data_type]
            print("{} Records with {} weight: {}".format(data_type, balancing_ratio, len(dataset_records)))

            dataset_weights.append(balancing_ratio)
            categorized_datasets.append(tf.data.TFRecordDataset(dataset_records).repeat())
            total_collected_samples += len(dataset_records)

        print("total_collected_samples: {} | len(self.tf_record_filenames): {}".format(total_collected_samples, len(self.tf_record_filenames)))

        if total_collected_samples != len(self.tf_record_filenames):
            print("Some datasets were not included after data type separation")
            print("total_collected_samples", total_collected_samples)
            print("len(self.tf_record_filenames)", len(self.tf_record_filenames))
        else:
            print("total_collected_samples == len(self.tf_record_filenames)")

        return categorized_datasets, dataset_weights

    def parse_single_record(self, record_fullname):
        self.dataset = tf.data.TFRecordDataset([record_fullname], num_parallel_reads=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.map(map_func=self.parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        input_image = None
        label = None
        for i, (input_images, labels) in enumerate(self.dataset):
            input_image = input_images
            label = labels

        return input_image, label

    def augment(self, image):
        random = np.random.rand()

        if random < 0.1:
            image_gray = tf.image.rgb_to_grayscale(image)
            aug_image = tf.image.grayscale_to_rgb(image_gray)
            aug_type = 'gray'

        elif random < 0.5:
            aug_image = tfio.experimental.color.rgb_to_bgr(image)
            aug_type = 'bgr'

        else:
            aug_image = image
            aug_type = None

        return aug_image, aug_type


def set_default_class(parsed_example: Dict, class_name: str, default_class_value: int, desirired_class_value: int) -> Dict:
    class_value = parsed_example[class_name]

    new_class_value = tf.cond(tf.equal(class_value, default_class_value),
                              lambda: tf.constant(desirired_class_value, dtype=class_value.dtype),
                              lambda: class_value)

    parsed_example[class_name] = new_class_value

    return parsed_example