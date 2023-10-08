import numpy as np
import pandas as pd

from settings import settings

def build_tp3_1():
    np.random.seed(1)

    num_points = 50

    points = np.random.uniform(0, 5, size=(2 * num_points, 2))
    labels = np.where(points[:, 1] >= points[:, 0], 1, -1)

    df = pd.DataFrame(data={'X': points[:, 0], 'Y': points[:, 1], 'Label': labels})

    df.to_csv(settings.Config.data_dir+'/TP3-1.csv', index=False)


def build_tp3_2():
    np.random.seed(3)

    num_points = 50

    points = np.random.uniform(0, 5, size=(2 * num_points, 2))
    labels = np.where(points[:, 1] >= points[:, 0], 1, -1)

    df = pd.DataFrame(data={'X': points[:, 0], 'Y': points[:, 1], 'Label': labels})

    num_misclassified_points = 10
    misclassified_x = np.random.uniform(0.5, 4.5, size=num_misclassified_points)
    misclassified_y = misclassified_x + np.random.choice([-0.5, 0.5], size=num_misclassified_points) * np.sqrt(1 - np.square(0.5))

    misclassified_labels = np.random.choice([-1, 1], size=num_misclassified_points)

    df_misclassified = pd.DataFrame(data={'X': misclassified_x, 'Y': misclassified_y, 'Label': misclassified_labels})

    df = pd.concat([df, df_misclassified])

    df.to_csv(settings.Config.data_dir+'/TP3-2.csv', index=False)



if __name__ == "__main__":
    build_tp3_1()

    build_tp3_2()