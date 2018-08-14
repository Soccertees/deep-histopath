from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

from datetime import datetime
import tensorflow as tf
from tensorflowonspark import TFNode
import logging

def print_log(worker_num, arg):
  logging.info("{0}: {1}".format(worker_num, arg))


class ExportHook(tf.train.SessionRunHook):
  def __init__(self, export_dir, input_tensor, output_tensor):
    self.export_dir = export_dir
    self.input_tensor = input_tensor
    self.output_tensor = output_tensor

  def end(self, session):
    logging.info("{} ======= Exporting to: {}".format(datetime.now().isoformat(), self.export_dir))
    signatures = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
        'inputs': {'image': self.input_tensor},
        'outputs': {'prediction': self.output_tensor},
        'method_name': tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      }
    }

    TFNode.export_saved_model(session,
                              self.export_dir,
                              tf.saved_model.tag_constants.SERVING,
                              signatures)
    logging.info("{} ====== Done exporting".format(datetime.now().isoformat()))


def map_fun(args, ctx, model_name="resnet_new", img_h=64, img_w=64, img_c=3):
  import numpy
  import time
  from train_mitoses import create_model, compute_data_loss, compute_metrics

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  # Delay PS node a bit, since workers seem to reserve GPUs quickly/reliably (w/o conflict)
  if job_name == "ps":
    time.sleep((worker_num + 1) * 5 )

  # Parameters
  batch_size = args.batch_size

  # Get TF cluster and server instances
  cluster, server = ctx.start_cluster_server(1, args.rdma)

  def feed_dict(batch):
    # Convert from [(labels, images)] to two numpy arrays of the proper type
    images = []
    labels = []
    for item in batch:
      images.append(item[1])
      labels.append(item[0])
    img_batch = numpy.array(images, dtype=numpy.float32)
    label_batch = numpy.array(labels, dtype=numpy.float32).reshape(-1, 1)

    return img_batch, label_batch

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    # Assigns ops to the local worker by default
    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:{0}".format(task_index),
      cluster=cluster)):

      # Placeholders or QueueRunner/Readers for input data
      with tf.name_scope('inputs'):
        images_var = tf.placeholder(tf.float32, (None, img_h, img_w, img_c), name="train_img")
        labels_var = tf.placeholder(tf.float32, (None, 1), name="train_label")
        tf.summary.image("train_img", images_var)

      with tf.name_scope('model'):
        model_tower, model_base = create_model(model_name, (img_h, img_w, img_c), images_var)
        model = model_tower
        logits = model.output
        probs = tf.nn.sigmoid(logits, name="probs")
        preds = tf.round(probs, name="preds")

      # with tf.name_scope('layer'):
      #   # Variables of the hidden layer
      #   with tf.name_scope('hidden_layer'):
      #     # Convolutional Layer #1
      #     conv1 = tf.layers.conv2d(
      #       inputs=images_var,
      #       filters=32,
      #       kernel_size=[5, 5],
      #       padding="same",
      #       activation=tf.nn.relu)
      #
      #     # Pooling Layer #1
      #     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
      #
      #     # Convolutional Layer #2 and Pooling Layer #2
      #     conv2 = tf.layers.conv2d(
      #       inputs=pool1,
      #       filters=64,
      #       kernel_size=[5, 5],
      #       padding="same",
      #       activation=tf.nn.relu)
      #     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
      #
      #     # Dense Layer
      #     pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
      #     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
      #     dropout = tf.layers.dropout(inputs=dense, rate=0.4)
      #
      #     # Logits Layer
      #     logits = tf.layers.dense(inputs=dropout, units=1)
      #     probs = tf.nn.sigmoid(logits, name="probs")
      #     preds = tf.round(probs, name="preds")

      global_step = tf.train.get_or_create_global_step()

      with tf.name_scope("loss"):
        with tf.control_dependencies([tf.assert_equal(tf.shape(labels_var)[0], tf.shape(preds)[0])]):
          loss = compute_data_loss(labels_var, logits)
          tf.summary.scalar("loss", loss)

      with tf.name_scope("train"):
        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

      with tf.name_scope("metrics"):
        num_thresholds = 11
        mean_loss, acc, ppv, sens, f1, pr, f1s, metric_update_ops, metric_reset_ops \
          = compute_metrics(loss, labels_var, preds, probs, num_thresholds)
        f1_max = tf.reduce_max(f1s)
        thresh_max = pr.thresholds[tf.argmax(f1s)]

        tf.summary.scalar("acc", acc)
        tf.summary.scalar("f1", f1)
        tf.summary.scalar("f1_max", f1_max)
        tf.summary.scalar("thresh_max", thresh_max)

      summary_op = tf.summary.merge_all()

    logdir = ctx.absolute_path(args.model)
    print("tensorflow model path: {0}".format(logdir))

    if job_name == "worker" and task_index == 0:
      summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())


    # The MonitoredTrainingSession takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(task_index == 0),
                                           checkpoint_dir=logdir,
                                           hooks=[tf.train.StopAtStepHook(last_step=args.steps)],
                                           chief_only_hooks=
                                           [ExportHook(ctx.absolute_path(args.export_dir),
                                                       images_var,
                                                       preds)]) as mon_sess:
      step = 0
      tf_feed = ctx.get_data_feed(args.mode == "train")
      while not mon_sess.should_stop() and not tf_feed.should_stop():
        fetch = tf_feed.next_batch(batch_size)
        logging.info("============= fetch type : {0}".format(type(fetch)))
        batch_imgs, batch_labels = feed_dict(fetch)
        feed = {images_var: batch_imgs, labels_var: batch_labels}

        if len(batch_imgs) > 0:
          if args.mode == "train":
            _, summary, step, metric_update, probs_output, preds_output, labels_output\
              = mon_sess.run([train_op, summary_op, global_step, metric_update_ops, probs, preds, labels_var],
                             feed_dict=feed)

            # print accuary and save model checkpoints to HDFS every 100 steps
            if (step % 10 == 0):
              logging.info("{0} step: {1} accuracy: {2} probs: {3} preds: {4} labels: {5}".format(
                datetime.now().isoformat(),
                step,
                mon_sess.run(acc),
                probs_output,
                preds_output,
                labels_output
                ))

            if task_index == 0:
              summary_writer.add_summary(summary, step)
          else:
            labels_output, preds_output, acc_output = mon_sess.run([labels_var, preds, acc], feed_dict=feed)
            results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p)
                       for l, p in zip(labels_output, preds_output)]
            tf_feed.bath_results(results)
            print("results: {0}, acc: {1}".format(results, acc_output))


      if mon_sess.should_stop() or step >= args.steps or len(batch_imgs) <= 0:
        tf_feed.terminate()

    # Ask for all the services to stop
    print("{0} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

  if job_name == "worker" and task_index == 0:
    summary_writer.close()














