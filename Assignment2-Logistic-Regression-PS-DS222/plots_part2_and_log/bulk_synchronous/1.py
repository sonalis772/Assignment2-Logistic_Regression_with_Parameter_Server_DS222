import re
import numpy as np
worker1=np.zeros((100))
i=0
for line in open('worker1'):
    match = re.search('cost= (\d+\.\d+)', line)
    if match:
        print (match.group(1),i)
        worker1[i]=match.group(1)
        i=i+1

i=0
worker2=np.zeros((100))
for line in open('worker2'):
    match = re.search('cost= (\d+\.\d+)', line)
    if match:
        print (match.group(1),i)
        worker2[i]=match.group(1)
        i=i+1

np.save("worker1",worker1)
np.save("worker2",worker2)


