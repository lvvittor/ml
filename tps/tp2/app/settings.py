from typing import Any
from pydantic import BaseSettings, BaseModel
from pathlib import Path
import json

def json_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
    """
    A simple settings source that loads variables from a JSON file
    at the project's root.
    """
    encoding = settings.__config__.env_file_encoding
    config_path = settings.__config__.config_path
    return json.loads(Path(config_path).read_text(encoding))


class KNN(BaseModel):
    weighted: bool
    k: int
    run_metrics: bool


class DecisionTree(BaseModel):
    run_metrics: bool


class Settings(BaseSettings):
    """
    Settings for the application.

    Settings are loaded from the following sources, in order:
    1. Environment variables
    2. JSON file at the project's root
    3. Secret environment variables

    For more information about parsing json files, see:

    https://jsontopydantic.com/
    """

    exercise: int = 1
    verbose: bool = False
    knn: KNN = KNN(weighted=False, k=5, run_metrics=False)
    decision_tree: DecisionTree = DecisionTree(run_metrics=False)

    class Config:
        env_file_encoding = 'utf-8'
        config_path = 'tps/tp2/config.json'
        data_dir = 'tps/tp2/data'
        out_dir = 'tps/tp2/out'

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                json_config_settings_source,
                env_settings,
                file_secret_settings,
            )


settings = Settings()