import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

post_dir=os.path.dirname(os.path.abspath( __file__))


fig=plt.figure()
a=np.load(post_dir+'/'+'decay_lr.npy')
b=np.load(post_dir+'/'+'inc_lr.npy')
c=np.zeros((150))+0.002
plt.plot(range(150),a,label='decaying_lr',marker='.',
     markersize=5,color ='b')
plt.plot(range(150),b,label='increasing_lr',marker='*',
     markersize=5,color ='r')
plt.plot(range(150),c,label='constant_lr',marker='.',
     markersize=5,color ='g')
plt.ylabel('learning rate')
plt.xlabel('number of epochs')
plt.legend()
fig.savefig('training_lr'+'.png')
plt.show()


fig=plt.figure()
a=np.load(post_dir+'/'+'decay_cost1.npy')
b=np.load(post_dir+'/'+'inc_cost1.npy')
c=np.load(post_dir+'/'+'cost1.npy')
plt.plot(range(150),a,label='loss_with_decaying_lr',marker='.',
     markersize=5,color ='b')
plt.plot(range(150),b,label='loss_with_increasing_lr',marker='*',
     markersize=5,color ='r')
plt.plot(range(150),c,label='loss_with_constant_lr',marker='.',
     markersize=5,color ='g')
plt.ylabel('cross entropy loss')
plt.xlabel('number of epochs')
plt.legend()
fig.savefig('training_loss'+'.png')
plt.show()

