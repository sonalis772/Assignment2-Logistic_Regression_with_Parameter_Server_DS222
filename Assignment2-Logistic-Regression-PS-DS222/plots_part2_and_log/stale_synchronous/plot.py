import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

post_dir=os.path.dirname(os.path.abspath( __file__))


fig=plt.figure()
a=np.load(post_dir+'/5/'+'worker1.npy')
b=np.load(post_dir+'/10/'+'worker1.npy')
c=np.load(post_dir+'/20/'+'worker1.npy')
print (a,b,c)
plt.plot(range(100),a,label='loss with staleness=5',marker='.',
     markersize=5,color ='b')
plt.plot(range(100),b,label='loss with staleness=10',marker='.',
     markersize=5,color ='r')
plt.plot(range(100),c,label='loss with staleness=20',marker='.',
     markersize=5,color ='g')
plt.ylabel('cross entropy loss with stale synchronous sgd')
plt.xlabel('number of epochs')
plt.legend()
fig.savefig('loss stale synchronous'+'.png')
plt.show()



