"""
Example Use:

python MAML.py \
  --data_dir /home/flash/vid2depth/data/bike \
  --num_tasks 4 \
  --num_inner_updates 3 \
  --inner_lr 0.0002 \
  --meta_lr 0.001 \
  --num_epochs 3

  """

import math
import os
import random
import tensorflow as tf
import os
import time
import reader
from absl import app
from absl import flags
from absl import logging
import modelMAML
import numpy as np
import util


# Maximum number of checkpoints to keep.
MAX_TO_KEEP = 100
INNER_LOOP = 0
OUTER_LOOP = 1
NUM_SCALES = 4
HOME_DIR = os.path.expanduser('~')
DEFAULT_DATA_DIR = os.path.join(HOME_DIR, '~/vid2depth/data/bike')
DEFAULT_CHECKPOINT_DIR = os.path.join(HOME_DIR, 'vid2depth/checkpoints')
gfile = tf.gfile

FLAGS = flags.FLAGS
# **** Base Model Flags ****
flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR, 'Preprocessed data.')
flags.DEFINE_float('beta1', 0.9, 'Adam momentum.')
flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')
flags.DEFINE_float('smooth_weight', 0.05, 'Smoothness loss weight.')
flags.DEFINE_float('ssim_weight', 0.15, 'SSIM loss weight.')
flags.DEFINE_float('icp_weight', 0.0, 'ICP loss weight.')
flags.DEFINE_integer('batch_size', 4, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')

# **** MAML Model Flags ****
flags.DEFINE_integer('num_tasks', 6, 'Number of tasks to train on.')
flags.DEFINE_integer('num_inner_updates', 2, 'Number of updates in the inner loop.')
flags.DEFINE_float('inner_lr', 0.0002, 'Learning rate for the inner loop.')
flags.DEFINE_float('meta_lr', 0.001, 'Learning rate for the meta-update.')
flags.DEFINE_integer('num_epochs', 4, 'Number of epochs(outer loop iterations) to train.')
flags.DEFINE_integer('train_steps', 2000, 'Number of training steps.')
flags.DEFINE_integer('summary_freq', 100, 'Save summaries every N steps.')

def initialize_model(inOrOut , subtask_dir):
    # Initialize your model here (adjust arguments as needed)
    maml_model = modelMAML.Model(data_dir = subtask_dir,
                                 for_inner = inOrOut,
                                 learning_rate=FLAGS.inner_lr,
                                 beta1=FLAGS.beta1,
                                 reconstr_weight=FLAGS.reconstr_weight,
                                 smooth_weight=FLAGS.smooth_weight,
                                 ssim_weight=FLAGS.ssim_weight,
                                 icp_weight=FLAGS.icp_weight,
                                 batch_size=FLAGS.batch_size,
                                 img_height=FLAGS.img_height,
                                 img_width=FLAGS.img_width,
                                 seq_length=FLAGS.seq_length,
                                 legacy_mode=FLAGS.legacy_mode) 

    # Get the trainable variables within the depth network scope
    depth_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='depth_net')

    return maml_model, depth_network_variables

def assign_weights(model, new_weights, sess):
    for var in tf.trainable_variables():
        if var.name in new_weights:
            sess.run(tf.assign(var, new_weights[var.name]))

def fetch_weights(train_model, checkpoint_dir, train_steps, summary_freq, initial_weights=None):
  """Train model."""
  vars_to_save = util.get_vars_to_restore()
  saver = tf.train.Saver(vars_to_save + [train_model.global_step],
                         max_to_keep=MAX_TO_KEEP)
  sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                           saver=None)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with sv.managed_session(config=config) as sess:
    if initial_weights is not None:
      assign_weights(train_model, initial_weights, sess)

    logging.info('Attempting to resume training from %s...', checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info('Last checkpoint found: %s', checkpoint)
    if checkpoint:
      saver.restore(sess, checkpoint)

    logging.info('Training...')
    start_time = time.time()
    last_summary_time = time.time()
    steps_per_epoch = train_model.reader.steps_per_epoch
    step = 1
    while step <= train_steps:
      fetches = {
          'train': train_model.train_op,
          'global_step': train_model.global_step,
          'incr_global_step': train_model.incr_global_step
      }

      if step % summary_freq == 0:
        fetches['loss'] = train_model.total_loss
        fetches['summary'] = sv.summary_op

      results = sess.run(fetches)
      global_step = results['global_step']

      if step % summary_freq == 0:
        sv.summary_writer.add_summary(results['summary'], global_step)
        train_epoch = math.ceil(global_step / steps_per_epoch)
        train_step = global_step - (train_epoch - 1) * steps_per_epoch
        this_cycle = time.time() - last_summary_time
        last_summary_time += this_cycle
        logging.info(
            'Epoch: [%2d] [%5d/%5d] time: %4.2fs (%ds total) loss: %.3f',
            train_epoch, train_step, steps_per_epoch, this_cycle,
            time.time() - start_time, results['loss'])

      if step % steps_per_epoch == 0:
        logging.info('[*] Saving checkpoint to %s...', checkpoint_dir)
        saver.save(sess, os.path.join(checkpoint_dir, 'model'),
                   global_step=global_step)

      # Setting step to global_step allows for training for a total of
      # train_steps even if the program is restarted during training.
      step = global_step + 1
    final_weights = {v.name: sess.run(v) for v in tf.trainable_variables()}
    return final_weights

def fetch_loss(train_model, compute_steps, summary_freq, initial_weights=None):
  """Compute average loss across the dataset."""
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sum_loss = 0

  with tf.Session(config=config) as sess:
    if initial_weights is not None:
      assign_weights(train_model, initial_weights, sess)
    
    steps_per_epoch = train_model.reader.steps_per_epoch
    step = 1
    logging.info('Computing Loss...')

    while step <= compute_steps:
      
      fetches = {
          'loss': train_model.total_loss,
          'incr_global_step': train_model.incr_global_step
      }

      results = sess.run(fetches)
      global_step = results['global_step']

      if step % summary_freq == 0:
        train_epoch = math.ceil(global_step / steps_per_epoch)
        train_step = global_step - (train_epoch - 1) * steps_per_epoch
        step_loss = results['loss']
        logging.info(
            'Epoch: [%2d] [%5d/%5d] loss: %.3f',
            train_epoch, train_step, steps_per_epoch, step_loss)

      # Setting step to global_step allows for computing loss for a total of
      # compute_steps even if the program is restarted during training.
      step = global_step + 1
      sum_loss += step_loss
      final_loss = sum_loss/step
    return final_loss

def maml_inner_loop(support_set_dir, weights, epoch_num, num_inner_updates):
    """
    Executes the MAML inner loop.
    Args:
    - support_set_dir: The directory of the support set.
    - weights: weights for the model
    - epoch_num: The number of the current epoch
    - num_inner_updates: The number of epochs fow which the inner loop must train on the support data

    
    Returns:
    - updated_weights: The model weights after inner loop updates.
    - support_set_loss: The support set loss after the final inner update.
    """
    model_copy, _ = initialize_model(INNER_LOOP, support_set_dir) # initialize with inner loop parameters
    checkpoint_dir_name = f'ckpts{epoch_num}'
    checkpoint_dir = os.path.join(support_set_dir, checkpoint_dir_name)
    if not gfile.Exists(checkpoint_dir):
        gfile.MakeDirs(checkpoint_dir)
    n_steps = reader.DataReader(support_set_dir, FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.seq_length, NUM_SCALES).steps_per_epoch
    # fetch weights after training on support data set
    support_set_weights = fetch_weights(model_copy, checkpoint_dir, num_inner_updates*n_steps, FLAGS.summary_freq, initial_weights = weights)
    
    # Return the weights after the inner loop updates and the final support set loss
    return support_set_weights

def maml_outer_loop(tasks, weights, num_inner_updates, epoch_num, meta_lr, num_tasks):
    """
    Executes the MAML outer loop, which includes the meta-update step.

    Args:
    - tasks: A list of tasks, each task containing a support set and a query set
    - weights: weights for the model
    - num_inner_updates: The number of updates in the inner loop
    - epoch_num: The number of the current epoch
    - meta_lr: The learning rate for the meta-update step
    - num_tasks: The total number of tasks in the taskspace selected

    Returns:
    - weights
    """
    total_query_set_loss = 0     
    for task in tasks:
        # Unpack the task data
        support_set_dir, query_set_dir = task
        # Run the inner loop to get updated weights
        updated_weights = maml_inner_loop(support_set_dir, weights, epoch_num, num_inner_updates) 
        model_copy, depth_network_variables = initialize_model(OUTER_LOOP, query_set_dir) # initialize with outer loop parameters 
        # Evaluate the model on the query set using the updated weights
        n_steps = reader.DataReader(query_set_dir, FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.seq_length, NUM_SCALES).steps_per_epoch
        query_set_loss = fetch_loss(model_copy, n_steps, FLAGS.summary_freq, initial_weights=updated_weights)
        total_query_set_loss += query_set_loss
    
    avg_query_set_loss = total_query_set_loss / num_tasks
    gradients = tf.gradients(avg_query_set_loss, depth_network_variables)
    # Apply meta-gradients (meta-optimizer step)
    # Apply meta-gradients (meta-optimizer step)
    meta_optimizer = tf.train.AdamOptimizer(learning_rate=meta_lr)
    update_ops = meta_optimizer.apply_gradients(zip(gradients, weights))

    # Ensure to run the update_ops within a TensorFlow session
    with tf.Session() as sess:
        sess.run(update_ops)
    return weights

def train_maml():
    
    """
    Train the MAML model.

    Args:
    - data_dir: Directory where tasks data is stored.
    - num_tasks: Number of tasks to train on.
    - num_inner_updates: Number of updates in the inner loop.
    - inner_lr: Learning rate for the inner loop.
    - meta_lr: Learning rate for the meta-update.
    - num_epochs: Number of epochs to train.
    """
    # Initialize weights for the first epoch
    initial_model, _ = initialize_model(INNER_LOOP, FLAGS.data_dir)
    weights = [weight for weight in tf.trainable_variables()]  # Get initial weights

    for epoch in range(FLAGS.num_epochs):
        tasks = []
        # Load tasks for this epoch
        for task_index in range(FLAGS.num_tasks):
            task_name = 'task{}'.format(task_index + 1)
            support_set_dir = os.path.join(FLAGS.data_dir, task_name, 'support')
            query_set_dir = os.path.join(FLAGS.data_dir, task_name, 'query')
            tasks.append((support_set_dir, query_set_dir))
        
        # Perform MAML outer loop
        updated_weights = maml_outer_loop(tasks, weights, FLAGS.num_inner_updates, epoch, FLAGS.inner_lr, FLAGS.meta_lr, FLAGS.num_tasks)
        # Update weights for the next epoch
        weights = updated_weights

def main(argv):
    del argv  # Unused.
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_maml()

if __name__ == '__main__':
    app.run(main)

