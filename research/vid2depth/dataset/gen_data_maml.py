from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
import multiprocessing
import numpy as np
import tensorflow as tf
import imageio
from absl import app
from absl import flags
from absl import logging
import dataset_loader

# Define flags for the command line arguments
flags.DEFINE_enum('dataset_name', None, ['kitti_raw_eigen', 'kitti_raw_stereo', 'kitti_odom', 'cityscapes', 'bike'], 'Dataset name.')
flags.DEFINE_string('dataset_dir', None, 'Location for dataset source files.')
flags.DEFINE_string('data_dir', None, 'Where to save the generated data.')
flags.DEFINE_integer('seq_length', 3, 'Length of each training sequence.')
flags.DEFINE_integer('img_height', 128, 'Image height.')
flags.DEFINE_integer('img_width', 416, 'Image width.')
flags.DEFINE_integer('num_threads', None, 'Number of worker threads. Defaults to number of CPU cores.')

flags.mark_flag_as_required('dataset_name')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('data_dir')

FLAGS = flags.FLAGS

def _generate_data():
    if not tf.io.gfile.exists(FLAGS.data_dir):
        tf.io.gfile.makedirs(FLAGS.data_dir)

    global dataloader  # pylint: disable=global-variable-undefined
    if FLAGS.dataset_name == 'bike':
        dataloader = dataset_loader.Bike(FLAGS.dataset_dir,
                                        img_height=FLAGS.img_height,
                                        img_width=FLAGS.img_width,
                                        seq_length=FLAGS.seq_length)
    elif FLAGS.dataset_name == 'kitti_odom':
        dataloader = dataset_loader.KittiOdom(FLAGS.dataset_dir,
                                            img_height=FLAGS.img_height,
                                            img_width=FLAGS.img_width,
                                            seq_length=FLAGS.seq_length)
    elif FLAGS.dataset_name == 'kitti_raw_eigen':
        dataloader = dataset_loader.KittiRaw(FLAGS.dataset_dir,
                                            split='eigen',
                                            img_height=FLAGS.img_height,
                                            img_width=FLAGS.img_width,
                                            seq_length=FLAGS.seq_length)
    elif FLAGS.dataset_name == 'kitti_raw_stereo':
        dataloader = dataset_loader.KittiRaw(FLAGS.dataset_dir,
                                            split='stereo',
                                            img_height=FLAGS.img_height,
                                            img_width=FLAGS.img_width,
                                            seq_length=FLAGS.seq_length)
    elif FLAGS.dataset_name == 'cityscapes':
        dataloader = dataset_loader.Cityscapes(FLAGS.dataset_dir,
                                            img_height=FLAGS.img_height,
                                            img_width=FLAGS.img_width,
                                            seq_length=FLAGS.seq_length)
    else:
        raise ValueError('Unknown dataset')
    all_frames = range(dataloader.num_train)
    frame_chunks = np.array_split(all_frames, 100)

    manager = multiprocessing.Manager()
    all_examples = manager.dict()
    num_threads = FLAGS.num_threads or multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_threads)
    np.random.seed(8964) # responsible for repeatability of train valid split

    for frame_chunk in frame_chunks:
        all_examples.clear()
        pool.map(_gen_example_star, zip(frame_chunk, itertools.repeat(all_examples)))
        
        # Here, we divide the work for 'support' and 'query' directories
        for count, (_, example) in enumerate(all_examples.items()):
            original_folder = example['folder_name']
            sub_dir = 'support' if count % 5 == 0 else 'query'
            example['folder_name'] = os.path.join(original_folder[:4], sub_dir, original_folder)

            # Make sure the new subdirectory structure exists
            save_dir = os.path.join(FLAGS.data_dir, example['folder_name'])
            if not tf.io.gfile.exists(save_dir):
                tf.io.gfile.makedirs(save_dir)

            # Save the example
            _save_example(example, save_dir)

    pool.close()
    pool.join()

def _gen_example_star(params):
    _gen_example(*params)

def _gen_example(i, all_examples):
    example = dataloader.get_example_with_index(i)
    if not example:
        return
    all_examples[i] = example

def _save_example(example, save_dir):
    image_seq_stack = _stack_image_seq(example['image_seq'])
    img_filepath = os.path.join(save_dir, '%s.jpg' % example['file_name'])
    imageio.imwrite(img_filepath, image_seq_stack.astype(np.uint8))

    # Save camera intrinsics
    intrinsics = example['intrinsics']
    cam_filepath = os.path.join(save_dir, '%s_cam.txt' % example['file_name'])
    cam_info = '%f,0.,%f,0.,%f,%f,0.,0.,1.' % (intrinsics[0, 0], intrinsics[0, 2], intrinsics[1, 1], intrinsics[1, 2])
    with tf.io.gfile.GFile(cam_filepath, 'w') as cam_f:
        cam_f.write(cam_info)

def _stack_image_seq(seq):
    return np.hstack(seq)

def main(_):
    _generate_data()

if __name__ == '__main__':
    app.run(main)

