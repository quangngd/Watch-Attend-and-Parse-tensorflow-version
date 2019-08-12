import sys
import os
import numpy
def cmp_result(label,rec):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def process(recfile, labelfile, resultfile):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    rec_mat = {}
    label_mat = {}
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            rec_mat[key] = latex
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            label_mat[key] = latex
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(label, rec)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    wer = float(total_dist)/total_label
    sacc = float(total_line_rec)/total_line

def process_top10(recfile, labelfile, resultfile):
    total_dist = numpy.zeros(10, dtype=float)
    total_label = numpy.zeros(10, dtype=float)
    total_line = numpy.zeros(10, dtype=float)
    total_line_rec = numpy.zeros(10, dtype=float)
    rec_mat ={}
    label_mat = {}
    top_i_acc = numpy.zeros(10, dtype=float)
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split('\t')
            key = tmp[0]
            latex = tmp[1].strip().split(' ')
            if latex[0] == '': latex = latex[1:]
            score = float(tmp[2])
            if key not in rec_mat:
                rec_mat[key] = list()
            rec_mat[key].append((latex, score))
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            label_mat[key] = latex
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec_mat[key_rec].sort(key=lambda x:x[1]) #sort by score (cost)
        rec = rec_mat[key_rec]
        res = [cmp_result(label, _rec[0]) for _rec in rec] # [(dist, llen)]
        total_dist += [_res[0] for _res in res]
        total_label += [_res[1] for _res in res]
        total_line += 1
        total_line_rec += [int(_res[0]==0) for _res in res]
        top_i_acc += [numpy.array([int(_res[0]==0) for _res in res][:i]).max() for i in range(1,11)]
    wer = total_dist/total_label
    sacc = total_line_rec/total_line
    top_i_acc /= total_line

    with open(resultfile,'w') as f_result:
        for i in range(10):
            f_result.write(f'At {i+1}th:\tWER:{wer[i]}\tExpRate:{sacc[i]}\n')
        for i in range(10):
            f_result.write(f'Top {i+1}:\tExpRate:{top_i_acc[i]}\n')
        


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ('compute-wer.py recfile labelfile resultfile')
        sys.exit(0)
    process_top10(sys.argv[1], sys.argv[2], sys.argv[3])