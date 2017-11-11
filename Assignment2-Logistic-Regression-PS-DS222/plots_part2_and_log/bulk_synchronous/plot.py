import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

post_dir=os.path.dirname(os.path.abspath( __file__))


fig=plt.figure()
a=np.load(post_dir+'/'+'worker1.npy')
b=np.load(post_dir+'/'+'worker2.npy')
print (a,b)
plt.plot(range(100),a,label='loss on worker1',marker='.',
     markersize=5,color ='b')
plt.plot(range(100),b,label='loss on worker2',marker='*',
     markersize=5,color ='r')
plt.ylabel('cross entropy loss with bulk synchronous sgd')
plt.xlabel('number of epochs')
plt.legend()
fig.savefig('loss bul synchronous'+'.png')
plt.show()



