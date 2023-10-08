import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from settings import settings

def main():
	match settings.exercise:
		case 1:
			exercise_1()
		case 2:
			exercise_2()
		case _:
			raise ValueError("Invalid exercise number")


def exercise_1():
    df = pd.read_csv(settings.Config.data_dir+'/TP3-1.csv')

    class1_df = df[df['Label'] == 1]
    class2_df = df[df['Label'] == -1]

    plt.scatter(class1_df['X'], class1_df['Y'], c='red', label='1')

    plt.scatter(class2_df['X'], class2_df['Y'], c='blue', label='-1')
	
    x_values = np.linspace(0, 5, 100)
    plt.plot(x_values, x_values, '--', c='green', label='y=x')

    plt.show()
	
    df = pd.read_csv(settings.Config.data_dir+'/TP3-2.csv')

    class1_df = df[df['Label'] == 1]
    class2_df = df[df['Label'] == -1]

    plt.scatter(class1_df['X'], class1_df['Y'], c='red', label='1')

    plt.scatter(class2_df['X'], class2_df['Y'], c='blue', label='-1')
	
    x_values = np.linspace(0, 5, 100)
    plt.plot(x_values, x_values, '--', c='green', label='y=x')

    plt.show()


def exercise_2():
    pass


if __name__ == "__main__":
    main()
