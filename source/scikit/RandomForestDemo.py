# coding=gbk
import numpy
import pandas
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold

from source.config.projectConfig import projectConfig
import matplotlib.pyplot as plt


class RandomForestDemo:

    @staticmethod
    def demo():
        data = []
        df_out = pandas.read_csv(projectConfig.getRandomForestTestData())
        print(df_out.shape)
        print(df_out.columns.values)
        matrix = df_out.dropna().sample(frac=1).as_matrix()
        print(type(matrix))
        label = matrix[:, 1]
        print(label.shape)
        print(label)
        category = pandas.Categorical(label)
        print(type(category.codes))
        label = category.codes
        features = matrix[:, 2:]
        print(type(features))
        print(features.shape)

        kf = KFold(n_splits=10)
        for train, test in kf.split(label):
            features_train = features[train]
            features_test = features[test]
            label_train = label[train]
            label_test = label[test]
            # # print(features_train)
            # # for index in train:
            # #     if features_train is None:
            # #         features_train = features[index, :]
            # #     else:
            # #         features_train = numpy.concatenate(features_train, features[index, :])
            # print(features_train.shape)

            features_train, features_test, label_train, label_test \
                = train_test_split(features, label, test_size=0.3, random_state=0)
            print(features_train.shape)
            print(features_test.shape)
            print(label_train.shape)
            print(label_test.shape)

            clf = RandomForestClassifier(n_estimators=2000, criterion='gini')
            clf.fit(features_train, label_train)
            predict_results = clf.predict_proba(features_test)[:, 1]
            print(predict_results)
            # print(predict_results.shape)
            # print(label_test)
            # print(accuracy_score(predict_results, label_test))
            # conf_mat = confusion_matrix(label_test, predict_results)
            # print(classification_report(label_test, predict_results))
            #
            fpr, tpr, thresholds = roc_curve(label_test, predict_results)
            print('threaholds：')
            print(thresholds)
            print(fpr)
            print(tpr)
            auc = roc_auc_score(label_test, predict_results)
            print(auc)

            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
            plt.title('$ROC curve$')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.show()


if __name__ == '__main__':
    RandomForestDemo.demo()
