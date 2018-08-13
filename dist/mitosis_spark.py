from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.sql import SparkSession
import logging
import argparse
from pyspark import SparkContext, SparkConf

from pyspark.ml.image import ImageSchema
from tensorflowonspark import TFCluster

from dist.utils import toNpArray

import dist.mitosis_dist as mitosis_dist


def main(args=None):

  spark = SparkSession \
    .builder \
    .appName("mitosis_spark") \
    .getOrCreate()
  sc = spark.sparkContext

  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1
  num_ps = 1
  logging.info("============= Num of executors: {0}".format(num_executors))

  # parse args
  parser = argparse.ArgumentParser()
  parser.add_argument("--appName", default="mitosis_spark", help="application name")
  parser.add_argument("--mitosis_img_dir", required=True, help="path to the mitosis image file")
  parser.add_argument("--normal_img_dir", required=True, help="path to the normal image file")

  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=16)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
  parser.add_argument("--export_dir", help="HDFS path to export saved_model",
                      default="mnist_export")
  parser.add_argument("--format", help="example format: (csv|pickle|tfr)",
                      choices=["csv", "pickle", "tfr"], default="csv")
  # parser.add_argument("--images", help="HDFS path to MNIST images in parallelized format")
  # parser.add_argument("--labels", help="HDFS path to MNIST labels in parallelized format")
  parser.add_argument("--model", help="HDFS path to save/load model during train/inference",
                      default="mnist_model")
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int,
                      default=num_executors)
  parser.add_argument("--output", help="HDFS path to save test/inference output",
                      default="predictions")
  parser.add_argument("--readers", help="number of reader/enqueue threads", type=int, default=1)
  parser.add_argument("--steps", help="maximum number of steps", type=int, default=99)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--mode", help="train|inference", default="train")
  parser.add_argument("--rdma", help="use rdma connection", default=False)
  args = parser.parse_args(args)


  # get mitosis images and labels
  # note that the numpy.ndarray could not be the key of RDD
  mitosis_img_df = ImageSchema.readImages(args.mitosis_img_dir, recursive=True)
  mitosis_train_rdd = mitosis_img_df.rdd.map(toNpArray).map(lambda img : (1, img))
  print("================", mitosis_train_rdd.count())

  # get normal images and labels
  normal_img_df = ImageSchema.readImages(args.normal_img_dir, recursive=True)
  normal_train_rdd = normal_img_df.rdd.map(toNpArray).map(lambda img: (0, img))
  print("================", normal_train_rdd.count())

  # get the train data set with mitosis and normal images
  data_RDD = mitosis_train_rdd.union(normal_train_rdd) #.repartition(args.cluster_size)

  print("================", data_RDD.count())

  sRDD = data_RDD.mapPartitions(lambda iter: [sum(1 for _ in iter)])

  for row in sRDD.collect():
    print("======================", row)


  cluster = TFCluster.run(sc, mitosis_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard,
                          TFCluster.InputMode.SPARK, log_dir=args.model)

  if args.mode == "train":
    cluster.train(data_RDD, args.epochs)
  else:
    labelRDD = cluster.inference(data_RDD)
    labelRDD.saveAsTextFile(args.output)


if __name__ == "__main__":
  main()


