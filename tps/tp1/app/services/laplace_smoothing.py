class LaplaceSmoothing:

    @staticmethod
    def smoothed_probability(word_counts, total_words, k=2):
        """
        Calculate Laplace smoothed probability
        
        :param word_counts: The number of occurrences.
        :param total_words: The total count of words.
        :param k: Smoothing factor (number of possible classes), default is 2.
        :return: Laplace smoothed probability.
        """
        return (word_counts + 1) / (total_words + k)