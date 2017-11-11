import re
import numpy as np
worker1_a=np.zeros((100))
i=0
lines = open('worker1').readlines()
print (lines[0])
j=0
for line in lines:
    match = re.search('cost= (\d+\.\d+)', line)
    if match:
        print (lines[j+1])
        worker1_a[i]=float(lines[j+1])
        i=i+1
    j=j+1

i=0
lines = open('worker2').readlines()
j=0
worker2_a=np.zeros((100))
for line in lines:
    match = re.search('cost= (\d+\.\d+)', line)
    if match:
        print (lines[j+1])
        worker2_a[i]=float(lines[j+1])
        i=i+1
    j=j+1
np.save("worker1_a",worker1_a)
np.save("worker2_a",worker2_a)
#print(worker2_a)


