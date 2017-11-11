'''
Distributed Tensorflow example of using data parallelism and share model parameters.


Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python asynchronous2.py --job_name="ps" --task_index=0 
pc-02$ python asynchronous2.py --job_name="ps" --task_index=1  
pc-03$ python asynchronous2.py --job_name="worker" --task_index=0 
pc-04$ python asynchronous2.py --job_name="worker" --task_index=1

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np
import h5py

# cluster specificationlocalhost:2222
parameter_servers = ["10.24.1.210:2225",
		    "10.24.1.211:2225"]
workers = [ "10.24.1.213:2225", 
      "10.24.1.214:2225"]




cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# config
batch_size = 4000
learning_rate = 0.002
training_epochs = 100
logs_path = "/home/sonalis772/project/assignments/assignment2/1"

label_train=np.load("train_l.npy").astype(np.float32)
#tfidf_documents_test=np.load("test.npy").astype(np.float32)
label_test=np.load("test_l.npy").astype(np.float32)
h5f1 = h5py.File('train.h5','r')
tfidf_documents_train = h5f1['d1'][:]
h5f2 = h5py.File('test.h5','r')
tfidf_documents_test = h5f2['d2'][:]
print("data_loaded")

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":

  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):
        global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False)
	learning_rate = 0.002
	training_epochs = 100
	batch_size = 4000
	display_step = 1

	# tf Graph Input
	x = tf.placeholder(tf.float32, [None, tfidf_documents_train.shape[1]]) 
	y = tf.placeholder(tf.float32, [None, 50]) 
	# Set model weights
	W = tf.Variable(tf.random_normal([tfidf_documents_train.shape[1], 50]))
	b = tf.Variable(tf.random_normal([50]))

	weight=tf.reshape(tf.reduce_sum(y,0)/tf.reduce_sum(tf.reduce_sum(y,0)),[1,50])

	# Construct model
	pred = (tf.matmul(x, W) + b) # Softmax
	weight_per_label = tf.transpose( tf.matmul(y
		                   , tf.transpose(weight)) ) 

	xent = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
	loss = tf.reduce_mean(xent) #shape 1
	regularizer = tf.nn.l2_loss(W)
	cost = tf.reduce_mean(loss + 0.01 * regularizer)
	# Gradient Descent
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	prediction = tf.nn.softmax(tf.matmul(x, W) + b)
        test_prediction = tf.nn.softmax(tf.matmul(tfidf_documents_test, W) + b)
	def accuracy(predictions, labels):
	    p=0
	    k=predictions.shape[0]
	    for i in range(predictions.shape[0]):
		if (labels[i,np.argmax(predictions[i,:])]!=0):
		   p=p+1
		   #print("correct")
		#else:
		   #print("wrong")
	    #print (np.argmax(predictions[50,:]),predictions[50,:],labels[50,:]) 
	    return (100.0 * p/ labels.shape[0])
	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
        tf.summary.scalar("cost", cost)
        tf.summary.scalar("accuracy", accuracy(test_prediction,label_test))
	# merge all summaries into a single "operation" which we can execute in a session 
	summary_op = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
	print("Variables initialized ...")
    

    
  save_cost1=np.zeros((training_epochs,1))
  save_cost2=np.zeros((training_epochs,1))
  sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

  begin_time = time.time()

  with sv.prepare_or_wait_for_session(server.target) as sess:
    
	    # create log writer object (this will log on every machine)
	    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
		
	    # perform training cycles
	    start_time = time.time()
	    for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(tfidf_documents_train.shape[0]/batch_size)
		# Loop over all batches
		for i in range(total_batch):
		    batch_xs=tfidf_documents_train[i*batch_size:i*batch_size+batch_size,:]
		    batch_ys = label_train[i*batch_size:i*batch_size+batch_size,:]
		    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
		                                                  y: batch_ys})
		    # Compute average loss
		    avg_cost += c / total_batch
		# Display logs per epoch step
		if (epoch+1) % display_step == 0:
		    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
		print (accuracy(sess.run(prediction, feed_dict={x: tfidf_documents_test}),label_test))
                if FLAGS.task_index == 0:
                   save_cost1[epoch,0]=avg_cost
                else:
                   save_cost2[epoch,0]=avg_cost
	    
	    print("Optimization Finished!")
	    print("Total Time: %3.2fs" % float(time.time() - start_time))
	    print ("train_accuracy=",accuracy(sess.run(prediction, feed_dict={x: tfidf_documents_train}),label_train))
            begin_time = time.time()
	    print ("test_accuracy=",accuracy(sess.run(prediction, feed_dict={x: tfidf_documents_test}),label_test))
            print("test time: %3.2fs" % float(time.time() - begin_time))
            np.save("asyn1",save_cost1)
            np.save("asyn2",save_cost2)

  sv.stop()
  print("done")
