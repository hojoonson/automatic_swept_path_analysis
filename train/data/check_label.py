with open('./train/trainlabels/Scherule_trainlabels.txt','r') as f:
    data=f.readlines()
    truecount=0
    falsecount=0
    for element in data:
        splited = element.split(' ')
        label = splited[3]
        if label=='0':
            falsecount+=1
        elif label=='1':
            truecount+=1
    print(f'True : {truecount}, False : {falsecount}')

