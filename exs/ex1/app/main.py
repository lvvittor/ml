import pandas as pd
from settings import settings
from services import PlotService


if __name__ == "__main__":
    df = pd.read_excel(f"{settings.Config.data_dir}/data.xlsx")
    df.columns = ["saturated_fats", "alcohol", "calories", "sex"]
    plot_service = PlotService(df)
    plot_service.boxplot()
    print(df.head())