import random
import math
import statistics
from collections import defaultdict
from algorithms import Null_Model_Predictor
from metrics import Evaluator

class Cross_Validator:
    """
    A class for cross validation tools

    Contains methods splitting data into partitions or folds. These get implemented by the cross validate
    function which is robust enough to handle both regression and classification and has two different modes
    of cross validation. The k-fold cross validation mode performs cross validation on k folds. The kx2-fold
    mode performs kx2 fold cross validation which follows a different process, and it also has an additional
    phase for hyperparameter tuning.

    Args:
        None

    Attributes:
        None
    
    Methods:
        kx2_fold_split(dataset, fold_mode='80,20', randomize=True): Splits the data either 50-50 or 80-20
        k_fold_split(dataset, k=10, randomize=True): Splits the data into k folds
        stratified_kx2_fold_split(dataset, y, randomize=True): Splits the data into two folds while ensuring stratification of class labels
    """
    def kx2_fold_split(self, dataset : list, fold_mode='80,20', randomize=True):
        """
        Splits the data either into an 80%-20% split or 50%-50% split

        Args:
            dataset (list): The source dataset to split
            fold_mode (str, optional): Fold mode to use - '50,50' or '80,20' (default)
            randomize (bool, optional): Shuffle the dataset before splitting

        Returns:
            fold_A (list): The first fold, which will be the 80% one if '80,20' mode is active
            fold_B  (list): The second fold, which will be the 20% one if '80,20' mode is active
        """
        #Shuffle the dataset
        if randomize == True:
            random.shuffle(dataset)

        #Initialize variables
        length = len(dataset)
        fold_A = []
        fold_B = []

        #Set the multipliers for fold size
        if fold_mode == '80,20':
            percent_A = 0.8
            percent_B = 0.2

        if fold_mode == '50,50':
            percent_A = 0.5
            percent_B = 0.5

        #Determine fold sizes
        fold_A_size = math.floor(length * percent_A)
        fold_B_size = math.floor(length * percent_B)

        #Populate the folds
        position = 0
        for i in range(fold_A_size):
            fold_A.append(dataset[position])
            position += 1
        for j in range(fold_B_size):
            fold_B.append(dataset[position])
            position += 1

        return fold_A, fold_B
    
    def k_fold_split(self, dataset : list, k=10, randomize=True):
        """
        Splits the data into k folds

        Args:
            dataset (list): The source dataset to split
            k (int, optional): The number of folds to create
            randomize (bool, optional): Shuffle the dataset before splitting

        Returns:
            k_folds (list): A list of lists, where each of the lists is one of k folds
        """
        #Shuffle the data
        if randomize == True:
            random.shuffle(dataset)
        
        #Initialize variables
        k_folds = []
        length = len(dataset)
        position = 0
        fold_size = length // k

        #Generate the folds by taking subsets of the source dataset
        for i in range(k):
            fold = []
            for j in range(fold_size):
                fold.append(dataset[position])
                position += 1
            k_folds.append(fold)

        #Print some information to validate the splitting process
        print("Dataset length=" + str(len(dataset)) + ", k=" + str(k) + ", fold size=" + str(fold_size))

        return k_folds
    
    def stratified_kx2_fold_split(self, dataset : list, y : str, randomize=True):
        """
        Splits the data into two partitions preserving stratification of class labels

        Args:
            dataset (list): The source dataset to split
            y (str): The name of the feature to base the stratification on
            randomize (bool, optional): Shuffle the dataset before splitting

        Returns:
            fold_A (list): First of two randomized, stratified folds
            fold_B (list): Second of two randomized, stratified folds
        """
        #Shuffle the data
        if randomize == True:
            random.shuffle(dataset)

        #Initialize variables
        fold_A = []
        fold_B = []
        fold_A_positions = []
        fold_B_positions = []
        y_class_positions = defaultdict(list)

        #Create dictionary associating y class labels with positions in the dataset
        position = 0
        for row in dataset:
            y_class_positions[row[y]].append(position)
            position += 1

        #Shuffle each list of class positions, then split into two folds
        for key in y_class_positions:
            current = y_class_positions[key]
            random.shuffle(current)
            #fold_A_positions.append(current[:len(current) // 2])
            for i in current[:len(current) // 2]:
                fold_A_positions.append(i)
            #fold_B_positions.append(current[len(current) // 2:])
            for i in current[len(current) // 2:]:
                fold_B_positions.append(i)

        for n in range(len(fold_A_positions)):
            fold_A.append(dataset[n])

        for m in range(len(fold_B_positions)):
            fold_B.append(dataset[m])

        #Validate the stratification process
        fold_A_class_freqs = defaultdict(list)
        fold_B_class_freqs = defaultdict(list)
        for row in fold_A:
            if not row[y] in fold_A_class_freqs:
                fold_A_class_freqs[row[y]] = 1
            else:
                fold_A_class_freqs[row[y]] += 1
        for row in fold_B:
            if not row[y] in fold_B_class_freqs:
                fold_B_class_freqs[row[y]] = 1
            else:
                fold_B_class_freqs[row[y]] += 1
        print("Splitting data and stratifying")
        print("Label frequencies:")
        print("Fold_A --> " + str(fold_A_class_freqs))
        print("Fold_B --> " + str(fold_B_class_freqs))

        return fold_A, fold_B
    
    def cross_validate(self, dataset, x, y, mode='regression', validation_mode='kx2-fold', k=5):
        """
        Performs cross validation for a machine learning algorithm

        Args:
            dataset (list): The dataset to use for machine learning
            x (list): The list of model input features
            y (str): The target feature
            validation_mode (str, optional): 'kx2-fold' or 'k-fold' cross validation modes
            k (int, optional): The number of folds to use in cross validation
        
        Returns:
            None
        """
        #Initialize metrics evaluation object
        metrics_evaluator = Evaluator()
        
        print("\nBegin validation using mode: " + validation_mode + " cross validation")

        # ----- kx2 fold cross validation mode -----
        if validation_mode == 'kx2-fold':
            model_1 = Null_Model_Predictor()
            model_2 = Null_Model_Predictor()
            model_1_scores = []
            model_2_scores = []

            # --- Hyperparameter tuning ---
            print("\nHyperparameter tuning")
            for i in range(k):
                print("\nIteration --> " + str(i + 1))
                #Create 80% and 20% split
                training_set, hyperparam_tuning_set = self.kx2_fold_split(dataset)
                #Create 50-50 split of the 80% partition
                if mode == 'regression':
                    subset_A, subset_B = self.kx2_fold_split(training_set, fold_mode='50,50')
                elif mode == 'classification':
                    subset_A, subset_B = self.stratified_kx2_fold_split(training_set, y)

                print("Tuning/validation set size=" + str(len(hyperparam_tuning_set)))
                print("Fold_A size=" + str(len(subset_A)))
                print("Fold_B size=" + str(len(subset_B)))
                
                #Extract y actuals from the hyperparameter tuning set (20%)
                y_actuals = []
                for row in hyperparam_tuning_set:
                    y_actuals.append(row[y])

                #Train models and make predictions on the test set
                model_1_predictions = model_1.predict(subset_A, hyperparam_tuning_set, y, mode)
                model_2_predictions = model_2.predict(subset_B, hyperparam_tuning_set, y, mode)

                #Calculate performance metrics for both models
                if mode == 'regression':
                    model_1_score = metrics_evaluator.mean_squared_error(model_1_predictions, y_actuals)
                    model_2_score = metrics_evaluator.mean_squared_error(model_2_predictions, y_actuals)
                elif mode == 'classification':
                    model_1_score = metrics_evaluator.classification_score(model_1_predictions, y_actuals)
                    model_2_score = metrics_evaluator.classification_score(model_2_predictions, y_actuals)
                model_1_scores.append(model_1_score)
                model_2_scores.append(model_2_score)
            
            #Summarize model performance
            print("")
            print("Model 1 Scores: " + str(model_1_scores))
            print("Model 2 Scores: " + str(model_2_scores))

            model_1_mean_score = statistics.mean(model_1_scores)
            model_2_mean_score = statistics.mean(model_2_scores)

            print("Model 1 mean score: " + str(model_1_mean_score))
            print("Model 2 mean score: " + str(model_2_mean_score))

            if mode == 'classification':
                if model_1_mean_score > model_2_mean_score:
                    selected_model = 1
                    print("Model 1 performance is superior")
                elif model_1_mean_score == model_2_mean_score:
                    print("Models have equal performance")
                else:
                    selected_model = 2
                    print("Model 2 performance is superior")
            
            # --- Model Validation with tuned parameters ---
            print("\nModel training and evaluation")
            
            tuned_model_scores = []

            for i in range(k):
                print("\nIteration --> " + str(i + 1))
                #Split the 80% partition 50-50
                if mode == 'regression':
                    set_A, set_B = self.kx2_fold_split(training_set, fold_mode='50,50')
                elif mode == 'classification':
                    set_A, set_B = self.stratified_kx2_fold_split(training_set, y)

                print("Fold_A size=" + str(len(set_A)))
                print("Fold_B size=" + str(len(set_B)))

                #Get the actuals column for each data partition
                y_actuals_A = []
                y_actuals_B = []
                for row in set_A:
                    y_actuals_A.append(row[y])
                for row in set_B:
                    y_actuals_B.append(row[y])

                #Train models and make predictions
                tuned_model_1 = Null_Model_Predictor()
                tuned_model_2 = Null_Model_Predictor()
                model_1_predictions = tuned_model_1.predict(set_A, set_B, y, mode)
                model_2_predictions = tuned_model_2.predict(set_B, set_A, y, mode)

                #Calculate performance metrics for both models
                if mode == 'regression':
                    model_1_score = metrics_evaluator.mean_squared_error(model_1_predictions, y_actuals_B)
                    model_2_score = metrics_evaluator.mean_squared_error(model_2_predictions, y_actuals_A)
                elif mode == 'classification':
                    model_1_score = metrics_evaluator.classification_score(model_1_predictions, y_actuals_B)
                    model_2_score = metrics_evaluator.classification_score(model_2_predictions, y_actuals_A)
                tuned_model_scores.append(model_1_score)
                tuned_model_scores.append(model_2_score)

            #Summarize scores
            print("\nTuned model scores: " + str(tuned_model_scores))
            tuned_model_mean_score = statistics.mean(tuned_model_scores)
            print("Final score: " + str(tuned_model_mean_score))
            print("")
        
        # ----- k fold cross validation mode -----
        elif validation_mode == 'k-fold':
            #Initialize model
            model = Null_Model_Predictor()
            model_scores = []

            #Split the dataset into k folds
            folds = self.k_fold_split(dataset, k)

            #Iterate k times
            for i in range(k):
                #Set the test set to be the current fold
                test_set = folds[i]

                #Set the training set to be all of the other folds combined
                training_set = []
                for j in range(len(folds)):
                    for row in folds[j]:
                        if i == j:
                            continue
                        training_set.append(row)
                
                print("\nIteration --> " + str(i+1))
                print("Training set size=" + str(len(training_set)))
                print("Test set size=" + str(len(test_set)))

                #Get y actuals
                y_actuals = []
                for row in test_set:
                    y_actuals.append(row[y])
                
                #Train model and make predictions
                predictions = model.predict(training_set, test_set, y, mode)

                #Calculate performance metrics
                if mode == 'regression':
                    model_score = metrics_evaluator.mean_squared_error(predictions, y_actuals)
                elif mode == 'classification':
                    model_score = metrics_evaluator.classification_score(predictions, y_actuals)
                model_scores.append(model_score)

            #Summarize results
            mean_score = statistics.mean(model_scores)
            print("\nModel scores: " + str(model_scores))
            print("Mean score: " + str(mean_score))
            print("")