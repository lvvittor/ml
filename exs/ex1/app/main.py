import pandas as pd
import random
from settings import settings
from services import PlotService, DataService

def get_prev_next_random(df: pd.DataFrame, row: pd.Series, column: str, index: int):
    """Returns a random number between the previous row value in the dataframe and the next one.
    """
    if row[column] == 999.99 and index > 0 and index + 1 < len(df):
        _prev = float(df.at[index - 1, column])
        _next = float(df.at[index + 1, column])
        return round(random.uniform(_prev, _next), 2)
    return None


def get_min_max_random(df: pd.DataFrame, row: pd.Series, column: str, index: int):
    if row[column] == 999.99:
        min_value = df[column].min()
        max_value = df[column][df[column] != 999.99].max()
        return round(random.uniform(min_value, max_value), 2)
    return None


if __name__ == "__main__":
    df = pd.read_excel(f"{settings.Config.data_dir}/data.xlsx")
    df.columns = ["saturated_fats", "alcohol", "calories", "sex"]
    data_service = DataService(df)
    data_service.replace_invalid_values(["alcohol", "saturated_fats"], [get_prev_next_random, get_min_max_random])

    plot_service = PlotService(df)
    plot_service.boxplot()
    plot_service.male_female_boxplot(["saturated_fats", "alcohol", "calories"], True)
    plot_service.calories_category_boxplot(["alcohol"], True)
    print(df)