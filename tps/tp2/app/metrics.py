import numpy as np
import matplotlib.pyplot as plt

from settings import settings

def calc_evaluation_measures(classes, confusion_matrix):
    """
    Calculate TP, TN, FP, FN, accuracy, precision, fp_rate, recall and f1_score
    for each class.
    """
    evaluation_measures = {}

    for category in classes:
        evaluation_measures[category] = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0,
            "precision": 0,
            "fp_rate": 0,
            "recall": 0,
            "f1_score": 0
        }

    for i in range(len(classes)):
        for j in range(len(classes)):
            category_actual = classes[i]
            category_predicted = classes[j]
            
            cell_value = confusion_matrix[i][j]
            
            if category_actual == category_predicted:
                evaluation_measures[category_actual]["tp"] += cell_value
                for category_other in classes:
                    if category_other != category_actual:
                        evaluation_measures[category_other]["tn"] += cell_value
            else:
                evaluation_measures[category_actual]["fn"] += cell_value
                evaluation_measures[category_predicted]["fp"] += cell_value
    
    for _, measures in evaluation_measures.items():
        TP = measures["tp"]
        TN = measures["tn"]
        FP = measures["fn"]
        FN = measures["fp"]
        measures["accuracy"] = (TP + TN)/(TP+TN+FP+FN)
        measures["precision"] = (TP)/(TP+FP)
        measures["fp_rate"] = (FP)/(TN+FP)
        measures["recall"] = (TP)/(TP+FN)
        measures["f1_score"] = 2*measures["precision"]*measures["recall"]/(measures["precision"]+measures["recall"])
    
    return evaluation_measures


class MetricsCalculator:

    def __init__(self):
        pass

    def accuracy(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)
    
    def confusion_matrix(self, y_test, y_pred, desired_value):
        cm = np.zeros((2, 2), dtype=int)

        tp = np.sum((y_test == desired_value) & (y_pred == desired_value))
        tn = np.sum((y_test != desired_value) & (y_pred != desired_value))
        fp = np.sum((y_test != desired_value) & (y_pred == desired_value))
        fn = np.sum((y_test == desired_value) & (y_pred != desired_value))

        cm[0, 0] = tp
        cm[0, 1] = fp
        cm[1, 0] = fn
        cm[1, 1] = tn

        return cm
    
    def plot_confusion_matrix(self, cm, algorithm):
        labels = ["Returns ", "Not returns"]
        plt.imshow(cm, cmap="YlOrRd")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.annotate(cm[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', color = "white" if cm[i, j] > 100 else "black")

        if(algorithm == "decision_tree"):
            plt.title("Confusion matrix of decision tree classifier")
        else:
            plt.title("Confusion matrix of random forest classifier")
        plt.xlabel("Predicted result")
        plt.ylabel("Expected result")
        if(algorithm == "decision_tree"):
            plt.savefig(f"{settings.Config.out_dir}/decision_tree_cm.png")
            plt.clf()
        elif(algorithm == "random_forest"):
            plt.savefig(f"{settings.Config.out_dir}/random_forest_cm.png")
            plt.clf()
    
    def precision(self, y_test, y_pred, desired_value):
        tp = np.sum((y_test == desired_value) & (y_pred == desired_value))
        fp = np.sum((y_test != desired_value) & (y_pred == desired_value))
        return tp / (tp + fp)
