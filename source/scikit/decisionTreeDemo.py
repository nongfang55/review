# coding=gbk
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class decisionTreeDemo:

    @staticmethod
    def demo():
        iris = load_iris()
        # clf = tree.DecisionTreeClassifier()
        # clf = clf.fit(iris.data, iris.target)
        # dot_data = tree.export_graphviz(clf, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("iris")

        # print(iris.target)
        iris.target[iris.target == 1], iris.target[iris.target == 2] = 0, 1
        # print(iris.target)
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
        model = LogisticRegression(solver='newton-cg', multi_class='ovr')
        model.fit(x_train, y_train)

        y_pre = model.predict_proba(x_test)
        print(y_pre)
        y_0 = list(y_pre[:, 1])
        print(y_0)

        fpr, tpr, thresholds = roc_curve(y_test, y_0)
        print('threaholds：')
        print(thresholds)
        print(fpr)
        print(tpr)
        auc = roc_auc_score(y_test, y_0)
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
    decisionTreeDemo.demo()
