from __future__ import division
import string
import math
import re
import numpy as np
from collections import defaultdict
from itertools import dropwhile
from nltk.corpus import stopwords
import h5py




def tokenize1(document):
	values = document.split("\t")
	label=values[0].rstrip().split(",")
	words = re.sub("\d+", "", values[1].rsplit('"',1)[0].split('"',1)[1])
	regex = r'(\w*)'
	list1=re.findall(regex,words)
	while '' in list1:
	     list1.remove('')
	list1=map(str.lower, list1)
        stop_words = set(stopwords.words('english'))
        list1 = [w for w in list1 if not w in stop_words]
        return list1

def labelize(document):
	values = document.split("\t")
	label=values[0].rstrip().split(",")
        return label

    


dy = defaultdict(int)
dz = defaultdict(int)
train_dir="/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt"
#train_dir="/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt"
filename_train = open(train_dir,"r") 
lines = filename_train.readlines()[3:]
#lines=[lines[165]]
for line in lines:
	values = line.split("\t")
	labels=values[0].rstrip().split(",")
	words = re.sub("\d+", "", values[1].rsplit('"',1)[0].split('"',1)[1])
	regex = r'(\w*)'
	list1=re.findall(regex,words)
	while '' in list1:
	   list1.remove('')
	list1=map(str.lower, list1)
        stop_words = set(stopwords.words('english'))
        list1 = [w for w in list1 if not w in stop_words]
	for word in list1:
	   dy[word] += 1
        for label in labels:
	   dz[label] += 1
	   
#dy=sorted(dy.iteritems(), key=lambda (k,v): (v,k))
#dz=sorted(dz.iteritems(), key=lambda (k,v): (v,k))



threshold_value=100

i=0
for k, v in dy.items():
    if v < threshold_value:
        del dy[k]
    i=i+1

i=0
for la in dy.keys():
    dy[la]=i
    i=i+1

i=0
for lz in dz.keys():
    dz[lz]=i
    i=i+1

print(len(dy))
print(len(dz))
#print(dy)
#print(dz)

train_dir="/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt"
#train_dir="/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_train.txt"
filename_train = open(train_dir,"r") 
lines_train = filename_train.readlines()

test_dir="/scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt"
#test_dir="/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_test.txt"
filename_test = open(test_dir,"r") 
lines_test = filename_test.readlines()

train=np.zeros((len(lines_train),len(dy)),dtype=np.float32)
test=np.zeros((len(lines_test),len(dy)),dtype=np.float32)
train_l=np.zeros((len(lines_train),len(dz)),dtype=np.float32)
test_l=np.zeros((len(lines_test),len(dz)),dtype=np.float32)

i=0
for line in lines_test:
    word=tokenize1(line)
    label=labelize(line)

    for l in label:
        test_l[i,dz[l]]=1
    for w in word:
        test[i,dy[w]]=1
    test_l[i,:]=test_l[i,:]/np.sum(test_l[i,:])
    i=i+1
    


i=0
for line in lines_train:
    word=tokenize1(line)
    label=labelize(line)

    for l in label:
        train_l[i,dz[l]]=1
    for w in word:
        train[i,dy[w]]=1
    train_l[i,:]=train_l[i,:]/np.sum(train_l[i,:])
    i=i+1
    
print(np.sum(train[0,:]))
print(train_l[0,:])
print(np.sum(test[0,:]))
print(test_l[0,:])

np.save("train_l",train_l)
#np.save("train",train)
#np.save("test",test)
np.save("test_l",test_l)
h5f1 = h5py.File('train.h5', 'w')
h5f1.create_dataset('d1', data=train)
h5f2 = h5py.File('test.h5', 'w')
h5f2.create_dataset('d2', data=test)
h5f1.close()
h5f2.close()





