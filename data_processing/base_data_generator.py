import os.path
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf


class BaseDataGenerator:
    def __init__(self, tf_record_filenames: List[str], batch_size: int = 4, data_format: str = 'channels_first',
                 stop_repeat: bool = False, training_labels_only: bool = False, batch_balancing_data_ratios_dict: Optional[Dict] = None,
                 shuffle: bool = True, training: bool = False, verbose: int = 0):
        self.tf_record_filenames = np.array(tf_record_filenames)
        self.batch_size = batch_size
        self.data_format = data_format
        self.stop_repeat = stop_repeat
        self.training_labels_only = training_labels_only
        self.batch_balancing_data_ratios_dict = batch_balancing_data_ratios_dict
        self.shuffle = shuffle

        self.training = training
        self.verbose = verbose

        if self.shuffle:
            np.random.shuffle(self.tf_record_filenames)

        # if training:
        #     self.make_dataset_encoder()

        self.reset()

    def reset(self):
        if self.batch_balancing_data_ratios_dict:
            separated_datasets, dataset_weights = self.separate_datasets_over_data()
            self.dataset = tf.data.experimental.sample_from_datasets(separated_datasets, dataset_weights)
        else:
            self.dataset = tf.data.TFRecordDataset(self.tf_record_filenames, num_parallel_reads=tf.data.AUTOTUNE)

        self.dataset = self.dataset.map(map_func=self.parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if self.batch_size > 0:
            self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)

        # if self.training:
        #     self.dataset = self.dataset.apply(tf.data.experimental.ignore_errors())

        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

        # if self.cache_path is not None:
        #     self.dataset = self.dataset.cache(self.cache_path)

        # if self.shuffle:
        #     self.dataset = self.dataset.shuffle(self.batch_size * 10)

        if not self.stop_repeat:
            self.dataset = self.dataset.repeat()

        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    def parse_fn(self, serialized_example):
        raise NotImplementedError('parse_fn method not implemented')

    @tf.function
    def generate(self):
        for record in self.dataset:
            yield record
        print("Done generating dataset")

    def generate_n(self, iterations):
        for record in self.dataset.take(iterations):
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

    def make_dataset_encoder(self):
        self.dataset_name_place = 0

        if self.dataset_gen == '1st_gen':
            self.dataset_name_place = 3
        elif self.dataset_gen == '2nd_gen':
            self.dataset_name_place = 4
        elif self.dataset_gen == '3rd_gen':
            self.dataset_name_place = 3

        dataset_names = [record_name.split(os.sep)[-self.dataset_name_place] for record_name in self.tf_record_filenames]
        unique_dataset_names = sorted(list(set(dataset_names)))
        print("unique_dataset_names", unique_dataset_names)

        # set_names = [record_name.split('/')[-2] for record_name in self.tf_record_filenames]
        # unique_set_names = list(set(set_names))
        # print("unique_set_names", unique_set_names)

        self.dataset_name_to_int = {dataset_name: i for i, dataset_name in enumerate(unique_dataset_names)}
        self.int_to_dataset_name = {i: dataset_name for dataset_name, i in self.dataset_name_to_int.items()}

        if self.dataset_gen == '1st_gen' or self.dataset_gen == '3rd_gen':
            self.set_to_int = {'train': 0, 'val': 1, 'test': 2}

        elif self.dataset_gen == '2nd_gen':
            category_names = [record_name.split(os.sep)[-2] for record_name in self.tf_record_filenames]
            unique_category_names = sorted(list(set(category_names)))
            print("unique_category_names", unique_category_names)

            set_num = 0
            self.set_to_int = {}
            for set_name in ['train', 'val', 'test']:
                for category_name in unique_category_names:
                    self.set_to_int["{}{}{}".format(set_name, os.sep, category_name)] = set_num
                    set_num += 1

        self.int_to_set = {i: set_name for set_name, i in self.set_to_int.items()}

        print("self.dataset_name_to_int", self.dataset_name_to_int)
        print("self.int_to_dataset_name", self.int_to_dataset_name)
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
        if self.dataset_gen == '1st_gen':
            dataset_name, set_name, record_index = record_name.split('/')
        elif self.dataset_gen == '2nd_gen':
            dataset_name, set_name, group_name, record_index = record_name.split('/')
            set_name = "{}{}{}".format(set_name, os.sep, group_name)
        elif self.dataset_gen == '3rd_gen':
            set_name, dataset_name, group_name, record_index = record_name.split('/')
            # set_name = "{}{}{}".format(set_name, os.sep, group_name)

        record_true_name, record_str = os.path.splitext(record_index)
        image_id, aug_index = record_true_name.split('-')
        image_id = int(image_id)
        aug_index = int(aug_index)

        dataset_name_int = self.dataset_name_to_int[dataset_name]
        set_name_int = self.set_to_int[set_name]

        return dataset_name_int, set_name_int, image_id, aug_index

    def encode_record_name_tf(self, record_name):
        linux_generated = tf.reduce_any(tf.strings.regex_full_match(record_name, ".*/.*"))
        record_name_split = tf.cond(linux_generated,
                                    lambda: tf.strings.split(record_name, '/'),
                                    lambda: tf.strings.split(record_name, '\\'))

        # record_name_split = tf.strings.split(record_name, os.sep)
        record_true_name = tf.strings.split(record_name_split[-1], '.')[0]
        record_true_name_split = tf.strings.split(record_true_name, '-')

        dataset_name_int = self.dataset_name_to_int_tf[record_name_split[0]]

        # print(record_name_split)
        if self.dataset_gen == '1st_gen':
            set_name = record_name_split[1]
        elif self.dataset_gen == '2nd_gen':
            set_name = tf.strings.join([record_name_split[1], record_name_split[2]], os.sep)
        elif self.dataset_gen == '3rd_gen':
            set_name = record_name_split[0]

        set_name_int = self.set_to_int_tf[set_name]

        image_id = tf.strings.to_number(record_true_name_split[0], tf.int64)
        crop_index = tf.strings.to_number(record_true_name_split[1], tf.int64)
        aug_index = tf.strings.to_number(record_true_name_split[2], tf.int64) if self.dataset_gen != '3rd_gen' else tf.convert_to_tensor(0, dtype=tf.int64)

        if self.dataset_gen != '3rd_gen':
            return dataset_name_int, set_name_int, image_id, crop_index, aug_index

        elif self.dataset_gen == '3rd_gen':
            print("record_name_split")
            print(record_name_split)
            print(set_name_int, dataset_name_int, image_id, crop_index, aug_index)

            return set_name_int, dataset_name_int, image_id, crop_index, aug_index

    def decode_record_name(self, dataset_name_int, set_name_int, image_id, crop_index, aug_index):
        if self.dataset_gen != '3rd_gen':
            if dataset_name_int in self.int_to_dataset_name:
                dataset_name = self.int_to_dataset_name[dataset_name_int]

            else:
                # print("NO {} in self.int_to_dataset_name".format(dataset_name_int))
                # print(self.int_to_dataset_name)
                dataset_name = str(dataset_name_int)

            set_name = self.int_to_set[set_name_int]

            record_index = "{:05d}-{}-{}.record".format(int(image_id), int(crop_index), int(aug_index))

        else:
            temp = dataset_name_int
            dataset_name_int = set_name_int
            set_name_int = temp

            if dataset_name_int in self.int_to_dataset_name:
                dataset_name = self.int_to_dataset_name[dataset_name_int]

            else:
                # print("NO {} in self.int_to_dataset_name".format(dataset_name_int))
                # print(self.int_to_dataset_name)
                dataset_name = str(dataset_name_int)

            set_name = self.int_to_set[set_name_int]

            record_index = "{:05d}-{}.record".format(int(image_id), int(crop_index))

        record_name = os.path.join(dataset_name, set_name, record_index)

        return record_name

    def separate_datasets_over_data(self):
        if abs(sum(self.batch_balancing_data_ratios_dict.values()) - 1) < 0.001:
            print("The sum of dataset weights does not sum to 1. Sum = {}".format(sum(self.batch_balancing_data_ratios_dict.values())))

        # separated_record_files = []
        dataset_weights = []
        categorized_datasets = []
        total_collected_samples = 0

        unique_dataset_names = set([record_name.split(os.sep)[-self.dataset_name_place] for record_name in self.tf_record_filenames])
        if len(self.batch_balancing_data_ratios_dict) <= len(unique_dataset_names):
            print("unique_dataset_names")
            print(unique_dataset_names)
            print()
            print("self.batch_balancing_data_ratios_dict.keys()")
            print(self.batch_balancing_data_ratios_dict.keys())

        for dataset_name, balancing_ratio in self.batch_balancing_data_ratios_dict.items():
            if self.dataset_gen == '1st_gen' or self.dataset_gen == '3rd_gen':
                dataset_records = [record_name for record_name in self.tf_record_filenames if record_name.split(os.sep)[-self.dataset_name_place] == dataset_name]
            else:
                dataset_records = [record_name for record_name in self.tf_record_filenames
                                   if "{}-{}".format(record_name.split(os.sep)[-self.dataset_name_place], record_name.split(os.sep)[-2]) == dataset_name]

            print("{:25s} len: {:6d} | weight: {:.4f} | iterations: {}".format(dataset_name, len(dataset_records), balancing_ratio,
                                                                               len(dataset_records) / (self.batch_size * balancing_ratio)))

            dataset_weights.append(balancing_ratio)
            categorized_datasets.append(tf.data.TFRecordDataset(dataset_records, num_parallel_reads=tf.data.AUTOTUNE).repeat())
            total_collected_samples += len(dataset_records)

        print("total_collected_samples: {} | len(self.tf_record_filenames): {}".format(total_collected_samples, len(self.tf_record_filenames)))

        if total_collected_samples != len(self.tf_record_filenames):
            print("Some datasets were not included after data type separation. Difference = {}".format(len(self.tf_record_filenames) - total_collected_samples))
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
        random = tf.random.uniform([])

        if random < 0.1:
            image_gray = tf.image.rgb_to_grayscale(image)
            aug_image = tf.image.grayscale_to_rgb(image_gray)
            aug_type = 'gray'

        elif random < 0.5:
            # aug_image = tfio.experimental.color.rgb_to_bgr(image)
            aug_image = tf.reverse(image, axis=[-1])
            aug_type = 'bgr'

        else:
            aug_image = image
            aug_type = ''

        return aug_image, aug_type

    def _parse_bboxes_and_landmarks(self, parsed_example, landmarks_shape=(10,)):
        bboxes_list = [tf.sparse.to_dense(parsed_example[bbox_key], default_value=-1) for bbox_key in BBOXES_KEYS]
        bboxes = tf.stack(bboxes_list, axis=1)

        landmarks_list = [tf.sparse.to_dense(parsed_example[landmark_key], default_value=-1) for landmark_key in LANDMARK_KEYS]
        landmarks = tf.stack([tf.reshape(landmark, (-1, 2)) for landmark in landmarks_list], axis=1)  # (landmark.shape[-1] // 2, 2)
        landmarks_flat = tf.reshape(landmarks, (-1, *landmarks_shape))

        return bboxes, landmarks_flat

    @staticmethod
    def set_default_class(parsed_example: Dict, class_name: str, default_class_value: int, desirired_class_value: int) -> Dict:
        class_value = parsed_example[class_name]

        new_class_value = tf.cond(tf.equal(class_value, default_class_value),
                                  lambda: tf.constant(desirired_class_value, dtype=class_value.dtype),
                                  lambda: class_value)

        parsed_example[class_name] = new_class_value

        return parsed_example