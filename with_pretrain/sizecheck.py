import numpy
import sys
import pickle as pkl
import gzip

fp=open("./pretrain_train.pkl",'rb')
features=pkl.load(fp, encoding='latin1')
fp.close()

fp2=open("./pretrain_train.txt",'r')
labels = fp2.read().strip().split('\n')
fp2.close()

batch_size= 4
batch_Imagesize=500000
maxImagesize=500000
dictionary=[]
with open("./dictionary.txt") as f:
    dictionary = f.read().strip().split('\n')

wToId = {}
for i, w in enumerate(dictionary):
    wToId[w] = i


targets={}
# map word to int with dictionary
for l in labels:
    tmp=l.strip().split('\t')
    uid=tmp[0]
    try:
        label = tmp[1]
    except IndexError:
        print(f"Index Error when reading label = tmp[1] at {l}")
        sys.exit()
    if label in dictionary:
        targets[uid]=numpy.zeros(len(dictionary))
        targets[uid][wToId[label]] = 1
    else:
        print('a word not in the dictionary !! sentence ',uid,'word ', w)
        sys.exit()

imageSize={}
for uid,fea in features.items():
    try:
        if(fea.shape[1]*fea.shape[2] > maxImagesize):
            print(f"image {uid}, size {fea.shape} bigger than {maxImagesize}, ignored")
        else:
            imageSize[uid]=fea.shape[1]*fea.shape[2]
    except IndexError as e:
        print(e)
        print(fea.shape)
        sys.exit()


imageSize= sorted(imageSize.items(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element


feature_batch=[]
label_batch=[]
feature_total=[]
label_total=[]
uidList=[]

batch_image_size=0
biggest_image_size=0
i=0
for uid,size in imageSize:
    if size>biggest_image_size:
        biggest_image_size=size
    fea=features[uid]
    lab=targets[uid]
    batch_image_size=biggest_image_size*(i+1)
    if size>maxImagesize:
        pass
        # print('image', uid, 'size ', size, ' bigger than', maxImagesize, 'ignore')
    else:
        uidList.append(uid)
        if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
            feature_total.append(feature_batch)
            label_total.append(label_batch)

            i=0
            biggest_image_size=size
            feature_batch=[]
            label_batch=[]
            feature_batch.append(fea)
            label_batch.append(lab)
            batch_image_size=biggest_image_size*(i+1)
            i+=1
        else:
            feature_batch.append(fea)
            label_batch.append(lab)
            i+=1

# last batch
feature_total.append(feature_batch)
label_total.append(label_batch)

print('total ',len(feature_total), 'batch data loaded')
