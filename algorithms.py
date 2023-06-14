import statistics

class Null_Model_Predictor:
    """
    A class representing a naive null model predictor 

    Learns from the training set and produces a predictions list in the same length as the test set. For classification,
    the model will always return the plurality, or most common, class label. For regression, the model will always return
    the mean of outputs in the training data.

    Args:
        None
    
    Attributes:
        None

    Methods:
        predict(training_set, test_set, y, mode): For classification, returns plurality. For regression, returns mean of outputs
    """
    def predict(self, training_set : list, test_set : list, y : str, mode='regression'):
        """
        Implements the null model prediction methods outlined above

        Args:
            training_set (list): Training data in the form of a list of dictionaries
            test_set (list): Test data in the form of a list of dictionaries
            y (str): The feature to predict
            mode (str, optional): Either 'classification' or 'regression' mode. Default is regression.

        Returns:
            predictions (list): A list the same length as the test set with the predicted labels or outputs
        """
        #Initialize variables
        predictions = []
        length = len(test_set)

        # ----- Regression mode -----
        #Compute column sum and apply this value to the entire predictions list
        if mode == 'regression':
            #sum = 0
            #for row in training_set:
            #    sum += row[y]
            y_column = []
            for row in training_set:
                y_column.append(row[y])
            mean = statistics.mean(y_column)
            for i in range(length):
                predictions.append(mean)

        # ----- Classification Mode -----
        #Compute the purality class label and apply this label to the entire predictions list
        if mode == 'classification':
            predictions = []
            class_frequencies = {}

            #Calculate frequency for each class label
            for row in training_set:
                current_label = row[y]
                if not current_label in class_frequencies:
                    class_frequencies[current_label] = 1
                elif current_label in class_frequencies:
                    class_frequencies[current_label] += 1
            
            #Determine which class label is most frequent
            max_freq = 0
            for key in class_frequencies:
                if class_frequencies[key] > max_freq:
                    max_freq = class_frequencies[key]
                    plurality = key

            for j in range(length):
                predictions.append(plurality)

        return predictions
