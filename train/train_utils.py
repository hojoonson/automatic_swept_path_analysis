from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             cohen_kappa_score, confusion_matrix, accuracy_score)
from tensorflow.keras.callbacks import Callback


class SkMetrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss = self.model.evaluate(self.validation_data[0], self.validation_data[1])[0]
        yhat_probs = self.model.predict(self.validation_data[0])
        yhat_probs = [x[0] for x in yhat_probs]
        yhat_classes = [0 if x < 0.5 else 1 for x in yhat_probs]
        testy = self.validation_data[1]
        print(yhat_classes[:30])
        print([x[0] for x in self.validation_data[1]][:30])
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(testy, yhat_classes)
        print('Accuracy: %f' % accuracy)
        print('Loss: %f' % loss)
        # precision tp / (tp + fp)
        precision = precision_score(testy, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(testy, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(testy, yhat_classes)
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(testy, yhat_classes)
        print('Cohens kappa: %f' % kappa)
        # ROC AUC
        auc = roc_auc_score(testy, yhat_probs)
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(testy, yhat_classes)
        print(matrix)
        # with open(csv_file_path, 'a') as f:
        #     csvwriter = csv.writer(f)
        #     csvwriter.writerow([accuracy, loss, precision, recall, f1, auc,
        #                        matrix[0], matrix[1], matrix[2], matrix[3]])


def lr_schedule(epoch):
    lr = 1e-3*0.5
    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 90:
        lr *= 1e-2
    elif epoch > 60:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
