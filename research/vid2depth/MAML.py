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


import tensorflow as tf
import os
from absl import app
from absl import flags
from absl import logging
from trainMaml import vid2depth

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/home/flash//vid2depth/data/bike', 'Directory where tasks data is stored.')
flags.DEFINE_integer('num_tasks', 5, 'Number of tasks to train on.')
flags.DEFINE_integer('num_inner_updates', 2, 'Number of updates in the inner loop.')
flags.DEFINE_float('inner_lr', 0.0002, 'Learning rate for the inner loop.')
flags.DEFINE_float('meta_lr', 0.001, 'Learning rate for the meta-update.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to train.')

def maml_inner_loop(support_set_dir, weights, epoch_num, num_inner_updates, inner_lr):
    """
    Executes the MAML inner loop.
    Args:
    - support_set_dir: The directory of the support set.
    - weights: A list of weights for the model
    - epoch_num: The number of the current epoch
    - num_inner_updates: The number of epochs fow which the inner loop must train on the support data
    - inner_lr: The learning rate for the inner loop updates.
    
    Returns:
    - updated_weights: The model weights after inner loop updates.
    - support_set_loss: The support set loss after the final inner update.
    """
    # Calculate the loss on the support set and perform the gradient updates
    support_set_loss, support_set_gradients = vid2depth(support_set_dir, True, weights[0], weights[1], weights[2], num_inner_updates, epoch_num)
    updated_weights = [
    tf.assign_sub(weights[0], inner_lr * support_set_gradients[0]),
    tf.assign_sub(weights[1], inner_lr * support_set_gradients[1]),
    tf.assign_sub(weights[2], inner_lr * support_set_gradients[2])
]
    # Return the weights after the inner loop updates and the final support set loss
    return updated_weights, support_set_loss


def maml_outer_loop(tasks, weights, num_inner_updates, epoch_num, inner_lr, meta_lr, num_tasks):
    """
    Executes the MAML outer loop, which includes the meta-update step.

    Args:
    - tasks: A list of tasks, each task containing a support set and a query set
    - weights: A list of weights, each row containes weights for a new epoch
    - num_inner_updates: The number of updates in the inner loop
    - epoch_num: The number of the current epoch
    - inner_lr: The learning rate for the inner loop updates
    - meta_lr: The learning rate for the meta-update step
    - num_tasks: The total number of tasks in the taskspace selected

    Returns:
    - None
    """

    total_outer_gradients = [tf.zeros_like(weight) for weight in weights]       
    for task in tasks:
        # Unpack the task data
        support_set_dir, query_set_dir = task

        # Run the inner loop to get updated weights
        updated_weights, _ = maml_inner_loop(support_set_dir, weights, epoch_num, num_inner_updates, inner_lr)

        # Evaluate the model on the query set using the updated weights
        _, query_set_gradients = vid2depth(query_set_dir, False, updated_weights[0], updated_weights[1], updated_weights[2], 1, epoch_num)
        
        for i in range(3):
            total_outer_gradients[i] = total_outer_gradients[i] + query_set_gradients[i]
    
    meta_gradients = [grad / num_tasks for grad in total_outer_gradients]

    # Apply meta-gradients (meta-optimizer step)
    meta_optimizer.apply_gradients(zip(meta_gradients, weights))

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
    weights = [tf.Variable(def_rec_wt, trainable=True),
     tf.Variable(def_sm_wt, trainable=True),
     tf.Variable(def_ssim_wt, trainable=True)]

    for epoch in range(FLAGS.num_epochs):
        tasks = []
        # Load tasks for this epoch
        for task_index in range(FLAGS.num_tasks):
            task_name = 'task{}'.format(task_index + 1)
            support_set_dir = os.path.join(FLAGS.data_dir, task_name, 'support')
            query_set_dir = os.path.join(FLAGS.data_dir, task_name, 'query')
            tasks.append((support_set_dir, query_set_dir))

        # Perform MAML outer loop
        maml_outer_loop(tasks, weights, FLAGS.num_inner_updates, epoch, FLAGS.inner_lr, FLAGS.meta_lr, FLAGS.num_tasks)


# Learning rates and hyperparameters for the MAML training
inner_lr = 0.0002
meta_lr = 0.001

def_rec_wt = 0.85
def_sm_wt = 0.05
def_ssim_wt = 0.15

meta_optimizer = tf.train.AdamOptimizer(meta_lr)

def main(argv):
    del argv  # Unused.
    train_maml()

if __name__ == '__main__':
    app.run(main)

