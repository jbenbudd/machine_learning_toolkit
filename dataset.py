import csv
import math
 
class Dataset:
    """
    A class representing a dataset

    This class reads comma-separated values from a text or csv file, and stores them inside of a list
    of dictionaries. Each row of data is represented by a dictionary, and each column is represented 
    by the dictionary's keys. Class methods include common data preprocessing functions. These methods 
    can be used to prepare a dataset for use in a machine learning algorithm.

    Args:
        data (list, optional): A list of dictionaries representing the data
    
    Attributes:
        _dataset (list): A list of dictionaries representing the source dataset
    
    Methods:
        load_csv(filepath, header): Imports the data from file and stores in _dataset attribute
        replace(source, target, column=None): Replaces values matching source with the value of target
        print(): Prints the dataset
        log_transform(column, log_base='e'): Performs a log transformation on the specified column with a custom log base
        convert_to_float(): Converts all numerical values in the dataset to floating point type
        impute(column, replace_value): Replaces all values matching replace_value with the column mean
        delete_column(column): Deletes the specified column
        delete_row(position): Deletes the row at the specified position
        add_column(column_name, column_value=None): Creates a new column with optional default values
        get_column(column): Returns the specified column
        get_dataset(): Returns the entire dataset as a list of dictionaries
        ordinal_encode(column, value_order_dict): Applies ordinal encoding assignments from a dictionary to a column
        one_hot_encode(column): One-hot encodes the specified column
        discretize(column, num_bins, mode='equal_width'): Discretizes the column (two modes: equal_frequency, equal_width)

    Example Usage:
        header = ["ID", "Type", "Time"]
        my_dataset = tools.Dataset("time_study.txt", header)
        my_dataset.one_hot_encode("Type")
        my_dataset.print()
    """

    def __init__(self, data=None):
        #Initialize dataset attribute, which will become a list of dictionaries
        self._dataset = []
        if not data == None:
            self._dataset = data
        self.convert_to_float()

    def load_csv(self, filepath : str, header : list):
        """
        Imports comma separated data from text or csv file

        Args:
            filepath (str): Path to the file containing data
            header (list): An ordered list of column names of type string

        Returns:
            None
        """
        with open(filepath, 'r') as csv_file:
            #Create the list of dictionaries from file
            reader = csv.DictReader(csv_file, fieldnames=header) 
            #Save the list of dictionaries to the _dataset class attribute
            for row in reader:
                self._dataset.append(row)
        self.convert_to_float()

    def replace(self, source, target, column=None):
        """
        Replaces values matching the source value with the target value

        Args:
            source (str or float): The value to look for in the dataset
            target (str or float): The value to replace the source values
            column (str, optional): Limits the replacement to just the specified column

        Returns:
            None
        """
        if not column == None:
            for row in self._dataset:
                if row[column] == source:
                    row[column] == target
        else:
            for row in self._dataset:
                for key, value in row.items():
                    if value == source:
                        row[key] = target

    def print(self, head=False):
        """
        Prints the entire dataset

        Args:
            head (bool, optional): Prints just the first few rows of data

        Returns:
            None
        """
        if head == False:
            for row in self._dataset:
                print(row)
        elif head == True:
            for i in range(15):
                print(self._dataset[i])        

    def log_transform(self, column : str, log_base=None):
        """
        Performs log transformation on the specified column with the specified log base

        Args:
            column (str): Name of the column to log transform
            log_base (int, optional): Log base value. Default base is 'e' for natural log

        Returns:
            None
        """
        if log_base == None:
            print("Performing log transformation on feature " + column + " with log base: e")
        else:
            print("Performing log transformation on feature " + column + " with log base: " + str(log_base))
        for row in self._dataset:
            if not log_base == None:
                row[column] = math.log(row[column], log_base)
            else:
                row[column] = math.log(row[column])

    def convert_to_float(self):
        """
        Converts all numerical values in the dataset to floating point type

        Args:
            None
        
        Returns:
            None
        """
        for row in self._dataset:
            for key, value in row.items():
                #Try to convert to float
                try:
                    row[key] = float(row[key])
                #Otherwise, skip
                except:
                    #print("Skipping " + row[key])
                    continue
    
    def impute(self, column : str, replace_value):
        """
        Imputes the missing values in the specified column with the column mean

        Args:
            column (str): Name of the column to impute
            replace_value (str or float): Value to replace, for example '?' or NaN

        Returns:
            None
        """
        #First compute the column mean by summing up values and dividing by length
        sum = 0
        length = 0
        for row in self._dataset:
            length += 1
            #Try to add the current value
            try:
                sum += row[column]
            #Skip the missing characters
            except:
                pass   
        mean = sum / length
        #Next, replace the missing values matching the replace_value parameter with the mean
        for row in self._dataset:
            if row[column] == replace_value:
                print("Replacing " + replace_value + " with " + column+ " feature mean: " + str(mean))
                row[column] = mean

    def delete_column(self, column):
        """
        Deletes the specified column

        Args:
            column (str): Name of the column to delete

        Returns:
            None
        """
        for row in self._dataset:
            del row[column]
    
    def delete_row(self, position):
        """
        Deletes the specified row

        Args:
            position (int): Position of the row to delete (first row = 0)

        Returns:
            None
        """
        self._dataset.pop(position)

    def add_column(self, column_name, column_value=None):
        """
        Adds a new column to the dataset

        Args:
            column_name (str): Name for the column header
            column_value (any type, optional): Value to populate the column, default is None
        
        Returns:
            None
        """
        for row in self._dataset:
            row[column_name] = column_value

    def get_column(self, column):
        """
        Gets the specified column from the dataset

        Args:
            column (str): Name of the column to get

        Returns:
            values (list): A list of the column values
        """
        values = []
        for row in self._dataset:
            values.append(row[column])
        return values
    
    def get_dataset(self):
        """
        Returns the dataset

        Args:
            None

        Returns:
            _dataset (list): A list of dictionaries representing the dataset
        """
        return self._dataset

    def ordinal_encode(self, column, value_order_dict):
        """
        Performs ordinal encoding on the specified column

        Args:
            column (str): Name of the column to ordinal encode
            value_order_dict (dictionary): A dictionary that assigns values with desired encoding

        Returns:
            None
        """
        #Iterate over the column
        for row in self._dataset:
            #Look up value in dict and replace with dict value
            if row[column] in value_order_dict:
                row[column] = value_order_dict[row[column]]
    
    def one_hot_encode(self, column):
        """
        One-hot encodes the specified column

        Args:
            column (str): Name of the column to one-hot encode

        Returns:
            None
        """
        #List of unique values
        unique_values = []
        #Dictionary mapping distinct values with new column names
        new_columns_names = {}

        #Iterate over the column and aggregate list of unique values
        for row in self._dataset:
            if row[column] not in unique_values:
                unique_values.append(row[column])

        #Populate dictionary with new column names, and create the new columns
        for value in unique_values:
            new_column_name = column + "_" + str(value).replace(".0","")
            self.add_column(new_column_name)
            new_columns_names[value] = new_column_name

        #Iterate over the dataset
        for row in self._dataset:
            #Iterate over the unique values list for each row
            for value in unique_values:
                #If the unique value is found in the column
                if row[column] == value:
                    #Set the unique value's column value to 1
                    row[new_columns_names[value]] = 1
                #If the unique value is not in the column
                else:
                    #Set the unique value's column value to 0
                    row[new_columns_names[value]] = 0
        #Delete the old column
        self.delete_column(column)

        #Rerun the float conversion to ensure all numerical values are of same type
        self.convert_to_float()

    def discretize(self, column : str, num_bins : int, mode='equal_width'):
        """
        Discretizes the selected column into the specified number of bins with 
        equal width or equal frequency. The resulting column values will each take 
        on a number from 0 to 1 - num_bins once complete.

        Args:
            column (str): Name of the column to discretize
            num_bins (int): Number of discrete bins to classify the values into
            mode (str, optional): equal_frequency or equal_width mode, default is equal_width

        Returns:
            None
        """
        print("Assigning " + column + " values to " + str(num_bins) + " discrete bins")
        #Get the column's values in a list and sort ascending
        column_values = self.get_column(column)
        column_values.sort()

        #Equal width mode
        if mode == 'equal_width':
            #First, get the range of the values
            first = column_values[0]
            last = column_values[-1]
            value_range = last - first
            #Then, get the width by dividing the range by number of bins
            width = value_range / num_bins

            #Create a list of the bin 'cutoffs', i.e. the values where each bin ends
            bin_cutoffs = []
            for i in range(num_bins):
                bin_cutoffs.append(i * width)

            #Iterate over the dataset
            for row in self._dataset:
                #Initialize bin assignment
                bin=0
                #Iterate over the number of bins until the correct bin is found for the current value
                for i in range(num_bins):
                    if row[column] < bin_cutoffs[i]:
                        break
                    elif row[column] > bin_cutoffs[i]:
                        bin = i
                        continue
                #Set the value equal to the bin number
                row[column] = bin + 1

        #Equal frequency mode
        if mode == 'equal_frequency':
            #First, determine the size of the bin by diving length of column by number of bins
            length = len(column_values)
            bin_size = int(length / num_bins)
            #Initialize bin assignments dictionary, which will map all values to a bin number
            bin_assignments = {}
            #Initialize iterator for use in iterating over the column
            position = 0

            #Iterate over the number of bins
            for i in range(num_bins):
                #Iterate over the size of a bin
                for j in range(bin_size):
                    #Create a dictionary entry for the current value, which will be the number of the current bin
                    bin_assignments[column_values[position]] = i
                    #Increment the column position
                    position += 1
            
            #Iterate over the dataset
            for row in self._dataset:
                #Assign the column's current value a bin number by looking it up in the dictionary 
                try:
                    row[column] = bin_assignments[row[column]] + 1
                except:
                    row[column] = num_bins
                    continue
