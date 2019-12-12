import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()
import time
import os, json
#compute 18
os.environ["TF_CONFIG"]  = json.dumps({
   "cluster": {
       "worker": ["192.168.9.16:1234",
           "192.168.9.17:1234",
           "192.168.9.18:1234",
           "192.168.9.23:1234",
           "192.168.9.24:1234",
           "192.168.9.25:1234",
           "192.168.9.26:1234",
           "192.168.9.27:1234",
           "192.168.9.28:1234",
           "192.168.9.29:1234",
           "192.168.9.30:1234",
           "192.168.9.31:1234",
           "192.168.9.32:1234",
           "192.168.9.105:1234",
           "192.168.9.106:1234",
           ],
       "chief": ["192.168.9.20:1235"],
       "ps": ["192.168.9.20:1236"],
       "evaluator": ["192.168.9.20:1237"]
   },
  "task": {"type": "worker", "index": 2}
})

BUFFER_SIZE = 10000
BATCH_SIZE = 32
EPOCHS = 10

def input_fn(mode, input_context=None):
    datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
    return mnist_dataset.map(scale).cache().repeat(EPOCHS).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
mnist_dataset = (datasets['train'])

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

mnist_dataset.map(scale).cache().repeat(EPOCHS).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
LEARNING_RATE = 1e-1

def model_fn(features, labels, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
  ])
    logits = model(features, training)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction="none",
      from_logits=True)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics_dict = {
            'accuracy': tf.metrics.accuracy(labels, tf.argmax(logits, axis=-1))
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics_dict)

    train_op = optimizer.minimize(
        loss, global_step=tf.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)

strategy = tf.distribute.experimental.ParameterServerStrategy()
# eval_strategy = tf.distribute.MirroredStrategy()
run_config = tf.estimator.RunConfig(
    experimental_distribute=tf.contrib.distribute.DistributeConfig(
       train_distribute=strategy,
       # eval_distribute=eval_strategy
    ),
   protocol="grpc+verbs"
   # save_checkpoints_steps=None,
    # save_checkpoints_secs=None
)
# In[206]:
estimator = tf.estimator.Estimator(config=run_config, model_fn=model_fn, model_dir='/home/rdma_for_ml/model_dir')

start = time.time()
tf.estimator.train_and_evaluate(
    estimator,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))
print(time.time()-start)

