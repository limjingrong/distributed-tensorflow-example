'''
Distributed Tensorflow 1.2.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time

# cluster specification
parameter_servers = ["localhost:1234"]
workers = [	"localhost:1235", "localhost:1236"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

# config
batch_size = 64
learning_rate = 0.02
training_epochs = 2
logs_path = "/tmp/mnist/1"

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

	# Between-graph replication
	with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):

		# count the number of updates
		global_step = tf.get_variable(
            'global_step',
            [],
            initializer = tf.constant_initializer(0),
			trainable = False)

		# input images
		with tf.name_scope('input'):
		  # None -> batch size can be any size, 784 -> flattened mnist image
		  x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
		  # target 10 output classes
		  y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

		# model parameters will change during training so we use tf.Variable
		tf.set_random_seed(1)
		with tf.name_scope("weights"):
			W1 = tf.Variable(tf.random_normal([784, 100]))
			W2 = tf.Variable(tf.random_normal([100, 10]))

		# bias
		with tf.name_scope("biases"):
			b1 = tf.Variable(tf.zeros([100]))
			b2 = tf.Variable(tf.zeros([10]))

		# implement model
		with tf.name_scope("softmax"):
			# y is our prediction
			z2 = tf.add(tf.matmul(x,W1),b1)
			a2 = tf.nn.sigmoid(z2)
			z3 = tf.add(tf.matmul(a2,W2),b2)
			y  = tf.nn.softmax(z3)

		# specify cost function
		with tf.name_scope('cross_entropy'):
			# this is our cost
			cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

		# specify optimizer
		with tf.name_scope('train'):
			# optimizer is an "operation" which we can execute in a session
			grad_op = tf.train.GradientDescentOptimizer(learning_rate)
			rep_op = tf.train.SyncReplicasOptimizer(
			    grad_op,
			    replicas_to_aggregate=len(workers),
			    total_num_replicas=len(workers),
				use_locking=True)
			train_op = rep_op.minimize(cross_entropy, global_step=global_step)
			
		init_token_op = rep_op.get_init_tokens_op()
		chief_queue_runner = rep_op.get_chief_queue_runner()

		with tf.name_scope('Accuracy'):
			# accuracy
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# create a summary for our cost and accuracy
		tf.summary.scalar("cost", cross_entropy)
		tf.summary.scalar("accuracy", accuracy)

		# merge all summaries into a single "operation" which we can execute in a session 
		summary_op = tf.summary.merge_all()
		init_op = tf.global_variables_initializer()
		print("Variables initialized ...")

	if FLAGS.task_index==0:
		local_init_op = rep_op.chief_init_op
	else:
		local_init_op = rep_op.local_step_init_op

	sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
		 recovery_wait_secs=1,
		 global_step=global_step,
		 local_init_op = local_init_op,
		 ready_for_local_init_op = rep_op.ready_for_local_init_op,
		 init_op=init_op)

	frequency = 100
	sess = sv.prepare_or_wait_for_session(server.target)
	# is chief
	if FLAGS.task_index == 0:
		sv.start_queue_runners(sess, [chief_queue_runner])
		sess.run(init_token_op)
	
	# create log writer object (this will log on every machine)
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	start_time = time.time()				
	# perform training cycles
	training_time = 0
	for epoch in range(training_epochs):
		print("Epoch", epoch)
		# number of batches in one epoch
		batch_count = int(mnist.train.num_examples/batch_size)
		epoch_start_time = time.time()
		for i in range(batch_count):
			print("Epoch", epoch, "batch", i)
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			# perform the operations we defined earlier on batch
			_, cost, summary, step = sess.run(
				[train_op, cross_entropy, summary_op, global_step], 
				feed_dict={x: batch_x, y_: batch_y})
			writer.add_summary(summary, step)

		training_time += time.time() - epoch_start_time	
		print("done training")

	total_time = time.time() - start_time
	wait_time = total_time - training_time
	print("Total Time: %3.2fs" % total_time)
	print("Training Time: %3.2fs" % float(training_time))
	print("Wait TIme: %3.2fs" % wait_time)
	test_start_time = time.time()
	print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	print("Training-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))

	time.sleep(5)
	#sv.request_stop()
	print("done")
