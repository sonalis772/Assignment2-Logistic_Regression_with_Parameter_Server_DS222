import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

post_dir=os.path.dirname(os.path.abspath( __file__))


fig=plt.figure()
a=np.load(post_dir+'/asynchronous/'+'worker1.npy')
b=np.load(post_dir+'/bulk_synchronous/'+'worker1.npy')
c=np.load(post_dir+'/stale_synchronous/10/'+'worker1.npy')

print (a,b)
plt.plot(range(100),a,label='train loss on asynchronous',marker='.',
     markersize=5,color ='b')
plt.plot(range(100),b,label='train loss on bulk_synchronous',marker='.',
     markersize=5,color ='r')
plt.plot(range(100),c,label='train loss on stale_synchronous',marker='.',
     markersize=5,color ='g')
plt.ylabel('A comparison of train loss with different sgd')
plt.xlabel('number of epochs')
plt.legend()
fig.savefig('combine1 loss'+'.png')
plt.show()



