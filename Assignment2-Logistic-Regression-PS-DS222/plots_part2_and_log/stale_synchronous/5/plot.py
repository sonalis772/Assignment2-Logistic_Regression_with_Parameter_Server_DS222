import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

post_dir=os.path.dirname(os.path.abspath( __file__))


fig=plt.figure()
a=np.load(post_dir+'/'+'st_syn2_5.npy')
plt.plot(range(100),a,label='loss',marker='.',
     markersize=5,color ='b')
plt.ylabel('cross entropy loss with bulk synchronous sgd')
plt.xlabel('number of epochs')
plt.legend()
fig.savefig('loss stale synchronous parameter 5'+'.png')
plt.show()



