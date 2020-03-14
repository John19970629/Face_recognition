# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:34:11 2019

@author: LO
"""

import numpy as np
from PIL import Image,ImageDraw

npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

fn = 0
ftable = []
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([0,y,x,h,w])
print(fn)                    
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19):
                    fn = fn + 1
                    ftable.append([1,y,x,h,w])
print(fn)
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19):
                    fn = fn + 1
                    ftable.append([2,y,x,h,w])
print(fn)
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([3,y,x,h,w])
print(fn)

def fe(sample,ftable,c):
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))
    if(ftype==0):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)
    elif(ftype==1):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y+h:y+h*2,x:x+w].flatten()
        output = np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx1],axis=1)
    elif(ftype==2):
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        idx3 = T[y:y+h,x+w*2:x+w*3].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)+np.sum(sample[:,idx3],axis=1)
    else:
        idx1 = T[y:y+h,x:x+w].flatten()
        idx2 = T[y:y+h,x+w:x+w*2].flatten()
        idx3 = T[y+h:y+h*2,x:x+w].flatten()
        idx4 = T[y+h:y+h*2,x+w:x+w*2].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx3],axis=1)+np.sum(sample[:,idx4],axis=1)
    return output

        
trpf = np.zeros((trpn,fn)) #2429X36648
trnf = np.zeros((trnn,fn)) #4548X36648
for c in range(fn):
    trpf[:,c] = fe(trainface,ftable,c)
    trnf[:,c] = fe(trainnonface,ftable,c)
    
def WC(pw,nw,pf,nf):
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
    polarity = 1
    if(error>0.5):
        polarity = 0
        error = 1 - error
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10):
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            polarity = 0
            error = 1 - error
        if(error<min_error):
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity
    
pw = np.ones((trpn,1))/trpn/2
nw = np.ones((trnn,1))/trnn/2


SC = []
for t in range(100):
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0
    for i in range(1,fn):
        me,mt,mp = WC(pw,nw,trpf[:,i],trnf[:,i])
        if(me<best_error):
            best_error = me
            best_feature = i
            best_theta = mt
            best_polarity = mp
    beta = best_error/(1-best_error)
    if(best_polarity == 1):
        pw[trpf[:,best_feature]>=best_theta]*=beta
        nw[trnf[:,best_feature]<best_theta]*=beta
    else:
        pw[trpf[:,best_feature]<best_theta]*=beta
        nw[trnf[:,best_feature]>=best_theta]*=beta
    alpha = np.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha])
    print(t)
    print(best_feature)

trps = np.zeros((trpn,1))
trns = np.zeros((trnn,1))
alpha_sum = 0
for i in range(100):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if(polarity==1):
        trps[trpf[:,feature]>=theta] += alpha
        trns[trnf[:,feature]>=theta] += alpha
    else:
        trps[trpf[:,feature]<theta] += alpha
        trns[trnf[:,feature]<theta] += alpha
trps /= alpha_sum
trns /= alpha_sum






I = Image.open('pic.jpg')
Igray = I.convert('L')
Iarr = np.array(Igray)
Igframe = np.zeros(((I.size[0]-18)*(I.size[1]-18),361))

xyindex = []
cunt = 0
for x in range (I.size[1]-18):
    for y in range(I.size[0]-18):
        Igframe[cunt,:] = Iarr[x:x+19,y:y+19].flatten()
        cunt+=1
        xyindex.append([x,y])

resulpf = np.zeros(((I.size[0]-18)*(I.size[1]-18),fn)) #41400X36648
        
for c in range(fn):
    resulpf[:,c] = fe(Igframe,ftable,c)
    
    
resulpa = np.zeros(((I.size[0]-18)*(I.size[1]-18),1))

alpha_sum = 0
for i in range(100):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if(polarity==1):
        resulpa[resulpf[:,feature]>=theta] += alpha
    else:
        resulpa[resulpf[:,feature]<theta] += alpha
resulpa /= alpha_sum

    
res = [] 
for idx in range(0, len(resulpa)) : 
    if resulpa[idx] > 0.6: 
        res.append(idx)

def draw(p):
    for i in range(len(p)):
        l = xyindex[res[i]]
        a = l[0]
        b = l[1]
        shape = [(b,a), (b+19,a+19)]
        rect = ImageDraw.Draw(I)
        rect.rectangle(shape,outline = "red")
    I.show()

draw(res)