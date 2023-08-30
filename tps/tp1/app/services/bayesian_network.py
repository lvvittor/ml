from typing import List
import pandas as pd
from services.laplace_smoothing import LaplaceSmoothing

class BayesianNetwork:

    def __init__(self, df, dag):
        self.df = df
        self.dag = dag

    def get_conditional_probability_table(self, prev_events, target_event):
        if not prev_events:
            # Calculate the standard probability for independent nodes
            return pd.DataFrame({target_event: self.df[target_event].value_counts(normalize=True)}).reset_index()
        else:
            joint_table = pd.crosstab(index=[self.df[idx] for idx in prev_events], columns=self.df[target_event], margins=True, normalize=True)
            conditional_prob_table = joint_table.div(joint_table['All'], level=0, axis=0).reset_index()
            return conditional_prob_table
    
    @property
    def network_tables(self):
        network_tables = {}
        for target_node in self.dag:
            parents = self.dag[target_node]
            prob_table = self.get_conditional_probability_table(prev_events=parents, target_event=target_node)
            network_tables[target_node] = prob_table
        return network_tables



    def get_filtered_probability(self, k: int = 2, **kwargs):
        filtered_df = self.df
        
        for key, value in kwargs.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        
        return LaplaceSmoothing.smoothed_probability(word_counts=len(filtered_df), total_words=len(self.df), k=k)
    
    def classify(self, gre, gpa, admit, rank):
        gpa_proba = self.get_filtered_probability(gpa_d=gpa, rank=rank)
        gre_proba = self.get_filtered_probability(gre_d=gre, rank=rank)
        admit_proba = self.get_filtered_probability(gre_d=gre, gpa_d=gpa, rank=rank, admit=admit)
        rank_proba = self.get_filtered_probability(k=4, rank=rank)

        return gpa_proba * gre_proba * admit_proba * rank_proba