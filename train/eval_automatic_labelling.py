import csv
import os
label_path = './automatic_labelling_result/v2_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-11 15:01:14.321387/result/label.csv'
# label_path = './automatic_labelling_result/v1_output:9_f4_transport_spmt_model_Custom_CNN_forimage_v2_checkpoint/2022-04-12 00:01:02.738618/result/label.csv'
with open(label_path, 'r', newline='') as labelfile:
    labels = csv.DictReader(labelfile, delimiter=',')
    tp=0; tn=0; fp=0; fn=0
    for element in labels:
        if element['label'] == '1' and element['gt'] == '1':
            tp +=1
        elif element['label'] == '0' and element['gt'] == '1':
            fn +=1
        elif element['label'] == '1' and element['gt'] == '0':
            fp +=1
        elif element['label'] == '0' and element['gt'] == '0':
            tn +=1
    print(f'TP: {tp}\tFN: {fn} \nFP: {fp}\tTN: {tn}')
    accuracy = round((tp+tn)/(tp+tn+fp+fn), 4)
    precision = round(tp/(tp+fp), 4)
    recall = round(tp/(tp+fn), 4)
    f1_score = round(2*precision*recall/(precision+recall),4)
    print(f'Accuracy : {accuracy}')
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'F1 Score : {f1_score}')