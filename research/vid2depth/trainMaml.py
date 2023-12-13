"""Train the model."""

# Example usage:
# vid2depth(data_dir = os.path.expanduser("~/vid2depth/data/bike/task1"),
#            reconstr_weight = 0.85, smooth_weight = 0.05, ssim_weight = 0.15, num_inner_updt = 1, epoch_num = 2)

import math
import os
import random
import time
import modelMaml as modelMaml
import numpy as np
import tensorflow as tf
import util

gfile = tf.gfile
# Maximum number of checkpoints to keep.
MAX_TO_KEEP = 100

def count_files(directory):
    file_count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                file_count += 1
    return file_count


def vid2depth(data_dir, reconstr_weight, smooth_weight, ssim_weight, num_iter, epoch_num, isInner):
  # Fixed seed for repeatability
  seed = 8964
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  data_dir = os.path.normpath(data_dir)
  checkpoint_dir_name = f'ckpts{epoch_num}'
  checkpoint_dir = os.path.join(data_dir, checkpoint_dir_name)
  if not gfile.Exists(checkpoint_dir):
    gfile.MakeDirs(checkpoint_dir)

  train_model = modelMaml.Model(data_dir, isInner, 0.0002, 0.9, reconstr_weight, smooth_weight, ssim_weight, 0, 4, 128, 416, 3, False)
  loss, grad = train(train_model, None, checkpoint_dir, num_iter*count_files(data_dir), 100)
  return loss, grad


def train(train_model, pretrained_ckpt, checkpoint_dir, train_steps,
          summary_freq):
  """Train model."""
  if pretrained_ckpt is not None:
    vars_to_restore = util.get_vars_to_restore(pretrained_ckpt)
    pretrain_restorer = tf.train.Saver(vars_to_restore)
  vars_to_save = util.get_vars_to_restore()
  saver = tf.train.Saver(vars_to_save + [train_model.global_step],
                         max_to_keep=MAX_TO_KEEP)
  sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                           saver=None)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  final_loss = None
  with sv.managed_session(config=config) as sess:
    if pretrained_ckpt is not None:
      pretrain_restorer.restore(sess, pretrained_ckpt)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
    if checkpoint:
      saver.restore(sess, checkpoint)

    print('Training...')
    start_time = time.time()
    last_summary_time = time.time()
    steps_per_epoch = train_model.reader.steps_per_epoch
    step = 1
    while step <= train_steps:
      fetches = {
          'train': train_model.train_op,
          'global_step': train_model.global_step,
          'incr_global_step': train_model.incr_global_step,
          # 'total_loss': train_model.total_loss
      }

      if step % summary_freq == 0:
        fetches['loss'] = train_model.total_loss
        fetches['summary'] = sv.summary_op
        fetches['gradients'] = train_model.gradients

      results = sess.run(fetches)
      global_step = results['global_step']

      if step % summary_freq == 0:
        sv.summary_writer.add_summary(results['summary'], global_step)
        train_epoch = math.ceil(global_step / steps_per_epoch)
        train_step = global_step - (train_epoch - 1) * steps_per_epoch
        this_cycle = time.time() - last_summary_time
        last_summary_time += this_cycle
        final_loss = results['loss']
        computed_gradients = fetches['gradients']  # Store the current loss
        print(f'Epoch: [{train_epoch:2d}] [{train_step:5d}/{steps_per_epoch:5d}] time: {this_cycle:4.2f}s ({int(time.time() - start_time)}s total) loss: {results["loss"]:.3f}')

      if step % steps_per_epoch == 0:
        print(f'[*] Saving checkpoint to {checkpoint_dir}...')
        saver.save(sess, os.path.join(checkpoint_dir, 'model'),
                   global_step=global_step)

      # Setting step to global_step allows for training for a total of
      # train_steps even if the program is restarted during training.
      step = global_step + 1

  return final_loss,computed_gradients

if __name__ == '__main__':
  print(" Dont run train.py script as main :) ")
