import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from settings import settings
from build_data import plot_datasets
from step_perceptron import StepPerceptron
from svm import SVM

def main():
	match settings.exercise:
		case 1:
			exercise_1()
		case 2:
			exercise_2()
		case _:
			raise ValueError("Invalid exercise number")


def exercise_1():
    plot_datasets()
	
    # ----------- A -------------
    df = pd.read_csv(settings.Config.data_dir+"/TP3-1.csv")
    inputs = df[['X', 'Y']].values
    expected_outputs = df['Label'].values
    
    step_perceptron = StepPerceptron(learning_rate=0.1, inputs=inputs, expected_outputs=expected_outputs)

    epochs, converged = step_perceptron.train(100)

    if settings.verbose:
        if not converged:
            print(f"Did not converge after {epochs} epochs\n")
        else:
            step_perceptron.save_animation()
            step_perceptron.save_animation_frames()
            print(f"Finished learning at {epochs} epochs")
        
    print(f"Perceptron weights (1): {step_perceptron.weights}")

    
    # ----------- B -------------

    df['HyperplaneDist'] = df.apply(lambda row: abs(step_perceptron.weights[0] + step_perceptron.weights[1] * row['X'] + step_perceptron.weights[2] * row['Y']), axis=1)

    df = df.sort_values(by='HyperplaneDist')

    # Select the 3 points closest to the hyperplane, making sure one of each label is selected
    selected_rows = []
    label_count = {}
    for _, row in df.iterrows():
        label = row['Label']
        label_count[label] = label_count.get(label, 0) + 1

        if label_count[label] <= 2:
            selected_rows.append(row)

        if len(selected_rows) == 3:
            break

    selected_df = pd.DataFrame(selected_rows)

    if settings.verbose:
        print(f"Points closest to the hyperplane:")
        print(selected_df.head())
    
    if len(selected_df[selected_df['Label'] == 1]) == 2:
        direction_points = selected_df[selected_df['Label'] == 1]
        translation_point = selected_df[selected_df['Label'] == -1]
    else:
        direction_points = selected_df[selected_df['Label'] == -1]
        translation_point = selected_df[selected_df['Label'] == 1]
    
    print(f"Direction points:")
    print(direction_points)

    print(f"Translation point:")
    print(translation_point)
    
    x1 = direction_points.iloc[0]['X']
    y1 = direction_points.iloc[0]['Y']
    x2 = direction_points.iloc[1]['X']
    y2 = direction_points.iloc[1]['Y']
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    x3 = translation_point["X"].iloc[0]
    y3 = translation_point["Y"].iloc[0]
    b2 = y3 - m * x3
    b = (b+b2)/2

    print(f"Distance to closest direction {abs(m * (x1) - y1 + b) / math.sqrt(m**2 + 1)}")
    print(f"Distance to closest translation {abs(m * (x3) - y3 + b) / math.sqrt(m**2 + 1)}")

    print(f"{b=}")

    x = range(0, 6)
    y = [m * (xi) + b for xi in x]

    plt.xlim([0, 5])
    plt.ylim([0, 5])

    class1_df = df[df['Label'] == 1]
    class2_df = df[df['Label'] == -1]

    plt.scatter(class1_df['X'], class1_df['Y'], c='red', label='1')

    plt.scatter(class2_df['X'], class2_df['Y'], c='blue', label='-1')
    plt.plot(x, y, label=f'y = {m}x + {b}', color='green', linestyle='-')

    plt.savefig(f"{settings.Config.out_dir}/optimal_hyperplane.png")

    plt.clf()

    
    # ----------- C -------------
    df = pd.read_csv(settings.Config.data_dir+"/TP3-2.csv")
    inputs = df[['X', 'Y']].values
    expected_outputs = df['Label'].values

    step_perceptron = StepPerceptron(learning_rate=0.1, inputs=inputs, expected_outputs=expected_outputs)

    epochs, converged = step_perceptron.train(50)

    if settings.verbose:
        if not converged:
            print(f"Did not converge after {epochs} epochs\n")
            step_perceptron.save_animation("step_perceptron_ejC")
        else:
            step_perceptron.save_animation("step_perceptron_ejC")
            step_perceptron.save_animation_frames("step_perceptron_ejC")
            print(f"Finished learning at {epochs} epochs")
    

    # ----------- D -------------

    runs = 10

    # -- Dataset 1 --

    perceptron_accuracies = []
    svm_accuracies = []

    df = pd.read_csv(settings.Config.data_dir+"/TP3-1.csv")

    inputs = df[['X', 'Y']].values
    expected_outputs = df['Label'].values

    for run in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(inputs, expected_outputs, test_size=0.2)

        # Step Perceptron

        step_perceptron = StepPerceptron(learning_rate=0.1, inputs=X_train, expected_outputs=y_train)

        epochs, converged = step_perceptron.train(50)

        y_hat = step_perceptron.predict(X_test)

        accuracy = accuracy_score(y_test, y_hat)

        perceptron_accuracies.append(accuracy)

        if run == 0:
            #step_perceptron.save_animation_frames(f"step_perceptron_test_1", count=1)
            step_perceptron.visualize_step_perceptron(X_test, y_test, f"step_perceptron_test_1")

        # SVM

        svm = SVM()

        svm.fit(X_train, y_train)

        y_hat = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_hat)

        svm_accuracies.append(accuracy)

        if run == 0:
            svm.visualize_svm(X_test, y_test, "svm_test_1")
            svm.visualize_svm(X_test, y_hat, "svm_hat_1")
    
    perceptron_accuracy = np.mean(perceptron_accuracies)
    perceptron_error = np.std(perceptron_accuracies)
    svm_accuracy = np.mean(svm_accuracies)
    svm_error = np.std(svm_accuracies)

    print(f"Step Perceptron Accuracy: {perceptron_accuracy}")
    print(f"SVM Accuracy: {svm_accuracy}")

    plt.bar(["Perceptron", "SVM"], [perceptron_accuracy, svm_accuracy], yerr=[perceptron_error, svm_error], capsize=5, color='skyblue', alpha=0.7)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{settings.Config.out_dir}/accuracy_comparison_1.png")
    plt.clf()

    # -- Dataset 2 --

    df = pd.read_csv(settings.Config.data_dir+"/TP3-2.csv")

    inputs = df[['X', 'Y']].values
    expected_outputs = df['Label'].values

    for run in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(inputs, expected_outputs, test_size=0.2)

        # Step Perceptron

        step_perceptron = StepPerceptron(learning_rate=0.1, inputs=X_train, expected_outputs=y_train)

        epochs, converged = step_perceptron.train(50)

        y_hat = step_perceptron.predict(X_test)

        accuracy = accuracy_score(y_test, y_hat)

        perceptron_accuracies.append(accuracy)

        if run == 0:
            #step_perceptron.save_animation_frames(f"step_perceptron_test_2", count=1)
            step_perceptron.visualize_step_perceptron(X_test, y_test, f"step_perceptron_test_2")

        # SVM

        svm = SVM()

        svm.fit(X_train, y_train)

        y_hat = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_hat)

        svm_accuracies.append(accuracy)

        if run == 0:
            svm.visualize_svm(X_test, y_test, "svm_test_2")
            svm.visualize_svm(X_test, y_hat, "svm_hat_2")
    
    perceptron_accuracy = np.mean(perceptron_accuracies)
    perceptron_error = np.std(perceptron_accuracies)
    svm_accuracy = np.mean(svm_accuracies)
    svm_error = np.std(svm_accuracies)

    print(f"Step Perceptron Accuracy: {perceptron_accuracy}")
    print(f"SVM Accuracy: {svm_accuracy}")

    plt.bar(["Perceptron", "SVM"], [perceptron_accuracy, svm_accuracy], yerr=[perceptron_error, svm_error], capsize=5, color='skyblue', alpha=0.7)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{settings.Config.out_dir}/accuracy_comparison_2.png")
    plt.clf()

def exercise_2():
    pass


if __name__ == "__main__":
    main()
