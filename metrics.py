class Evaluator:
    """
    A class for performance evaluation metrics

    Args:
        None

    Attributes:
        None

    Methods:
        classification_score(y_predicted, y_actual): Calculates proportion of correct class labels
        mean_squared_error(y_predicted, y_actual): Calculates the MSE for regression
    """
    def classification_score(self, y_predicted : list, y_actual : list):
        """
        Calculates proportion of correct class labels

        Args:
            y_predicted (list): The predicted labels in list form
            y_actual (list): The actual, or ground truth, labels in list form

        Returns:
            score (float): The percentage of class labels that were correct, in decimal form
        """
        #Initialize variables
        correct = 0
        incorrect = 0
        total = len(y_predicted)

        #Calculate number correct
        for i in range(total):
            if y_predicted[i] == y_actual[i]:
                correct += 1
            else:
                incorrect += 1
        
        #Get accuracy score by dividing number correct by total
        score = correct / total
        return score
    
    def mean_squared_error(self, y_predicted : list, y_actual : list):
        """
        Calculates the MSE for regression outputs

        Args:
            y_predicted (list): The predicted outputs in list form
            y_actual (list): The actual outputs in list form

        Returns:
            mse (float): The mean squared error between predicted and actuals
        """
        sum_squared_errors = 0
        total = len(y_predicted)
        for i in range(len(y_predicted)):
            squared_error = (y_predicted[i] - y_actual[i]) ** 2
            sum_squared_errors += squared_error
        mse = sum_squared_errors / total
        return mse
