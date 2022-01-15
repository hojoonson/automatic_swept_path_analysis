f = open("scherule_trainlabels.txt","r")
data=f.readlines()
truecount=0
falsecount=0
for element in data:
    if element.split(" ")[1]=="0":
        falsecount+=1
    elif element.split(" ")[1]=="1":
        truecount+=1
    print("True : "+str(truecount)+" False : "+str(falsecount))
f.close()

