import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split

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

    df = pd.read_csv(settings.Config.data_dir+"/TP3-2.csv")

    inputs = df[['X', 'Y']].values
    expected_outputs = df['Label'].values
	
    test_size = 0.3

    X_train, X_test, y_train, y_test = train_test_split(inputs, expected_outputs, test_size=test_size, random_state=40)
	
    svm = SVM()
	
    w, b, losses = svm.fit(X_train, y_train)
	
    svm.visualize_svm(X_test, y_test, w, b)


def exercise_2():
    pass


if __name__ == "__main__":
    main()
