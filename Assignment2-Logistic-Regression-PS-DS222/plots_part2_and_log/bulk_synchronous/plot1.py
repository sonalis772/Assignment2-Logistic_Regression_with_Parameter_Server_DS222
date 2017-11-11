import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

post_dir=os.path.dirname(os.path.abspath( __file__))


fig=plt.figure()
a=np.load(post_dir+'/'+'worker1_a.npy')
b=np.load(post_dir+'/'+'worker2_a.npy')
print (a,b)
plt.plot(range(100),a,label='accuracy on worker1',marker='.',
     markersize=5,color ='b')
plt.plot(range(100),b,label='accuracy on worker2',marker='*',
     markersize=5,color ='r')
plt.ylabel('test accuracy with bulk synchronous sgd')
plt.xlabel('number of epochs')
plt.legend()
fig.savefig('accuracy bulk synchronous'+'.png')
plt.show()



