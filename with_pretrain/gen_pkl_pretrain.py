import re
import pickle
import numpy as np
import os
from PIL import Image

labelPaths = [
    "../../CROHME2016/Task-2-Symbols-pngs/iso_GT_train.txt",
    "../../CROHME2016/Task-2-Symbols-pngs/iso_GT_val.txt",
]
imgDirs = [
    "../../CROHME2016/Task-2-Symbols-pngs/Task-2-Symbols-pngs/task2-trainSymb2014/trainingSymbols",
    "../../CROHME2016/Task-2-Symbols-pngs/Task-2-Symbols-pngs/task2-validation-isolatedTest2013b"
]
imgFromIdFuncs = [
    lambda id: f"iso{id}.png",
    lambda id: f"BOTH{id}.png"
]
labelRegexs = [
    ".+_(\d+),(.+)",
    ".+_(\d+),(.+)"
]
savePaths = [
    "./pretrain_train",
    "./pretrain_valid"
]


labelPaths = [
    "../../CROHME2016/Task-2-Symbols-pngs/iso_GT_val.txt",
]
imgDirs = [
    "../../CROHME2016/Task-2-Symbols-pngs/Task-2-Symbols-pngs/task2-validation-isolatedTest2013b"
]
imgFromIdFuncs = [
    lambda id: f"BOTH{id}.png"
]
labelRegexs = [
    ".+_(\d+),(.+)"
]
savePaths = [
    "./pretrain_debug"
]

"""
Format:
- Train files:
    - Labels: .+_(\d+),(.+)
        group 1: id,
        group 2: label
    e.g:
        101_Frank_150,i
        101_Frank_151,\theta
    - Img: iso{id}.png

- Valid files:
    - Labels: .+_(\d+),(.+)
        group 1: id,
        group 2: label
    e.g:
        2013_RIT_CROHME_rit_42190_1_11121, junk
        2013_RIT_CROHME_rit_42190_2_11122,1
    - Img: BOTH{id}.png
"""
dictionary = set()
dictionarySavePath = "./dictionary.txt"

hh = 128/3.0
"""
CROHME 2016 size = None * 128 (w,h)
"""

for labelPath, imgDir, imgFromIdFunc, labelRegex, savePath in zip(labelPaths, imgDirs, imgFromIdFuncs, labelRegexs, savePaths):
    features = {}
    labels = {}
    filesNotFound = []
    cnt = 0
    juncnt = 0
    with open(labelPath, "r") as labelFile:
        while 1:
            line = labelFile.readline().strip()
            if not line: break
            else:
                match = re.match(labelRegex,line)
                if not match:
                    raise Exception(f"Syntax error in {line}")
                else:
                    key = match.group(1)
                    label = match.group(2)
                    if "junk" in label:
                        juncnt+=1
                        continue
                    try:
                        im = Image.open(os.path.join(imgDir,imgFromIdFunc(key))).convert('L')
                        _im = im
                        resize_ratio = hh/im.size[1]
                        resized_size = int(im.size[0]*resize_ratio), int(im.size[1]*resize_ratio)
                        im = im.resize(resized_size)
                        if(im.size[1] != int(hh)):
                            print("Resize fail")
                            print(im.size)
                            sys.exit()
                        if(im.size[0] * im.size[1] > 500000):
                            print(f"Warning: pic {key} have size {im.size}")
                        im =  np.array(im)
                    except FileNotFoundError:
                        filesNotFound.append(imgFromIdFunc(key))
                        continue
                    features[key] = im[None, :, :]
                    labels[key] = label
                    
                    if label not in dictionary:
                        dictionary.add(label)
                    
                    cnt += 1
                    if cnt%1000==0:
                        print(f"Loaded {cnt} im")
    print(f"Done. Loaded {cnt} im.")
    print(f"Skip {juncnt} junks")
    print(f"Skip {','.join(filesNotFound)} file not found")

    with open(savePath + ".pkl", "wb") as saveFile:
        pickle.dump(features, saveFile)
        print('Save pkl done')

    with open(savePath + ".txt", "w") as saveFile:
        for k, v in labels.items():
            saveFile.write(f"{k}\t{v}\n")
        print('Save txt done')
                    
with open(dictionarySavePath, "w") as saveFile:
    for w in dictionary:
        saveFile.write(f"{w}\n")
    print('Save dict done')