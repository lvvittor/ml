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