import pandas as pd
from services.laplace_smoothing import LaplaceSmoothing

class AdmissionsPredictor:

    def __init__(self, df):
        self.df = AdmissionsPredictor.discretize_features(df)

    @classmethod
    def discretize_features(cls, df: pd.DataFrame):
        df['gpa_class'] = (df['gpa'] >= 3.0).astype(bool)
        df['gre_class'] = (df['gre'] >= 500).astype(bool)
        return df
    
    def get_filtered_probability(self, k: int = 2, **kwargs):
        filtered_df = self.df
        
        for key, value in kwargs.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        
        return LaplaceSmoothing.smoothed_probability(word_counts=len(filtered_df), total_words=len(self.df), k=k)
    
    def classify(self, gre, gpa, admit, rank):
        gpa_proba = self.get_filtered_probability(gpa_class=gpa, rank=rank)
        gre_proba = self.get_filtered_probability(gre_class=gre, rank=rank)
        admit_proba = self.get_filtered_probability(gre_class=gre, gpa_class=gpa, rank=rank, admit=admit)
        rank_proba = self.get_filtered_probability(k=4, rank=rank)

        return gpa_proba * gre_proba * admit_proba * rank_proba