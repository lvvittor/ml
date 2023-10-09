import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import settings

def build_tp3_1():
    np.random.seed(2)

    num_points = 200

    points = np.random.uniform(0, 5, size=(num_points, 2))
    labels = np.where(points[:, 1] >= points[:, 0], 1, -1)

    df = pd.DataFrame(data={'X': points[:, 0], 'Y': points[:, 1], 'Label': labels})

    df.to_csv(settings.Config.data_dir+'/TP3-1.csv', index=False)


def build_tp3_2():
    np.random.seed(5)

    num_points = 200

    points = np.random.uniform(0, 5, size=(2 * num_points, 2))
    labels = np.where(points[:, 1] >= points[:, 0], 1, -1)

    df = pd.DataFrame(data={'X': points[:, 0], 'Y': points[:, 1], 'Label': labels})

    num_misclassified_points = 40
    misclassified_x = np.random.uniform(0.5, 4.5, size=num_misclassified_points)
    misclassified_y = misclassified_x + np.random.uniform(-0.5, 0.5, size=num_misclassified_points) * np.sqrt(1 - np.square(0.5))

    misclassified_labels = np.random.choice([-1, 1], size=num_misclassified_points)

    df_misclassified = pd.DataFrame(data={'X': misclassified_x, 'Y': misclassified_y, 'Label': misclassified_labels})

    df = pd.concat([df, df_misclassified])

    df.to_csv(settings.Config.data_dir+'/TP3-2.csv', index=False)


def plot_datasets():
    df = pd.read_csv(settings.Config.data_dir+'/TP3-1.csv')

    class1_df = df[df['Label'] == 1]
    class2_df = df[df['Label'] == -1]

    plt.scatter(class1_df['X'], class1_df['Y'], c='red', label='1')

    plt.scatter(class2_df['X'], class2_df['Y'], c='blue', label='-1')
	
    x_values = np.linspace(0, 5, 100)
    plt.plot(x_values, x_values, '--', c='green', label='y=x')

    plt.savefig(f"{settings.Config.out_dir}/TP3-1.png")
    plt.close()
	
    df = pd.read_csv(settings.Config.data_dir+'/TP3-2.csv')

    class1_df = df[df['Label'] == 1]
    class2_df = df[df['Label'] == -1]

    plt.scatter(class1_df['X'], class1_df['Y'], c='red', label='1')

    plt.scatter(class2_df['X'], class2_df['Y'], c='blue', label='-1')
	
    x_values = np.linspace(0, 5, 100)
    plt.plot(x_values, x_values, '--', c='green', label='y=x')

    plt.savefig(f"{settings.Config.out_dir}/TP3-2.png")
    plt.close()


if __name__ == "__main__":
    build_tp3_1()

    build_tp3_2()