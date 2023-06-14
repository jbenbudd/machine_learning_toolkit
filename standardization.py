import statistics

class Z_Score_Standardizer:
    """
    A class that handles z score standardization

    Args:
        None

    Attributes:
        None

    Methods:
        z_score_standardize(training_set, test_set, feature): Performs z score standardization
    """
    def  z_score_standardize(self, training_set : list, test_set : list, feature : str):
        """
        Normalizes feature around the mean using z-score standardization. Z score
        calculated only on the training set, but standardization applied to both
        training and test sets.

        Args:
            training_set (list): Training data in the form of list of dictionaries
            test_set (list): Test data in the form of list of dictionaries
            feature (str): The name of the feature(column) to standardize

        Returns:
            training_set (list): The standardized training set
            test_set (list): The standardized test set
        """
        #Get the column to standardize from the training set
        print("Standardizing feature column: " + feature )
        column = []
        for row in training_set:
            column.append(row[feature])
        
        #Compute mean and standard deviation
        mean = statistics.mean(column)
        std_dev = statistics.stdev(column)

        print("Mean=" + str(mean))
        print("Standard deviation=" + str(std_dev))

        #For each row, compute z score for the feature and set feature to z score
        for row in training_set:
            z_score = (row[feature] - mean) / std_dev
            row[feature] = z_score
        for row in test_set:
            z_score = (row[feature] - mean) / std_dev
            row[feature] = z_score

        #Return the standardized sets
        return training_set, test_set