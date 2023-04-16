import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sklearn
from sklearn import preprocessing
import seaborn as sns
print(tf.__version__)


# This code imports several Python libraries that are commonly used in machine learning tasks including:

# - `pandas`: a data manipulation library that provides useful data structures for working with structured data such as `DataFrame`.
# - `numpy`: a library for numerical computing that provides high-performance array objects and vectorization capabilities.
# - `tensorflow`: an open-source machine learning framework for building and training deep neural networks.
# - `matplotlib`: a plotting library for creating static, animated, and interactive visualizations in Python.
# - `%matplotlib inline`: a magic command that allows you to display Matplotlib graphics inline within a Jupyter notebook or JupyterLab.
# - `sklearn.metrics`: a library that provides tools for model evaluation such as calculating explained variance score, mean absolute error, and other regression performance metrics.
# - `train_test_split`: a function from `sklearn.model_selection` which allows for splitting the dataset into training and test sets.
# - `r2_score`: a function from `sklearn.metrics` which calculates the coefficient of determination, or R-squared, for evaluating model performance.
# - `keras.models`: a high-level API for building and training neural networks in Keras.
# - `keras.layers`: a module containing built-in layers for building neural networks in Keras.
# - `keras.optimizers`: a module containing optimizer functions for configuring the training process in Keras.
# - `keras.callbacks`: a module containing callback functions for monitoring and controlling the training process in Keras.
# - `sklearn`: The Scikit-learn library provides a range of tools for supervised learning, including for classification, regression and clustering.
# - `preprocessing`: The scikit-learn preprocessing module provides functions for pre-processing input data such as scaling, normalization, & encoding categorical variables.
# - `seaborn`: a data visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

# Finally, the code prints the current version of TensorFlow being used in the system by calling the `tf.version` function. The `print` function is used to output the version information to the console.

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# read in the csv data into a pandas data frame and set the date as the index
df = pd.read_csv('/content/JaipurFinalCleanData.csv').set_index('date')

# This code will read a CSV file named 'JaipurFinalCleanData.csv' into a pandas DataFrame called `df`. The `set_index` method is then called on the DataFrame to set the 'date' column as the index. Here is what each part of the code does:

# - `pd.read_csv`: This function reads a file in CSV format and returns a DataFrame. The file path to the CSV file is specified as an argument between the parentheses. In this case, the file is located at '/content/JaipurFinalCleanData.csv'.
# - `.set_index('date')`: This is an operation that sets the 'date' column as the index of the DataFrame. The resulting DataFrame has a new index based on the values in the 'date' column. This column will now be used to reference and retrieve rows from the DataFrame.
# - The entire expression `pd.read_csv('/content/JaipurFinalCleanData.csv').set_index('date')` returns the resulting DataFrame with the 'date' column set as the index. The DataFrame is assigned to the variable name `df`, which can now be used to manipulate or analyze the data.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# execute the describe() function and transpose the output so that it doesn't overflow the width of the screen
df.describe().T

# This code is used to generate a descriptive summary of the DataFrame `df`. The `describe()` function returns a summary of the count, mean, standard deviation, minimum, and maximum values of each numeric column in the DataFrame. Here is what each part of the code does:

# - `df.describe()`: This method computes summary statistics of the DataFrame columns (excluding `object` type columns) and returns a new DataFrame containing the results. The resulting DataFrame includes count, mean, standard deviation, minimum, and maximum values for each numeric column in `df`.
# - `.T`: This is a transposition operation that changes the orientation of the DataFrame by flipping the rows and columns. This is done so that the output doesn't exceed the width of the screen when displayed in a Jupyter notebook or other terminal environments.
# - The entire expression `df.describe().T` computes the summary statistics for `df` and transposes the resulting DataFrame so that the rows and columns are flipped. The resulting DataFrame is a summary of the count, mean, standard deviation, minimum, and maximum values for each numeric column in `df`.

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sns.set(style="darkgrid")
for col in df.columns:
    sns.displot(df[col], kde=False)
    plt.title(f"Distribution of {col}")
    plt.show()

# This code generates a set of subplots that show the distribution of each column in the DataFrame `df`. Here is what each part of the code does:

# - `sns.set(style="darkgrid")`: This method sets the default Seaborn theme to "darkgrid", which is a black background with a grid overlay.
# - `for col in df.columns:`: This `for` loop iterates over each column in the DataFrame `df`. The `df.columns` attribute returns the list of column names in the DataFrame.
# - `sns.displot(df[col], kde=False)`: This method creates a histogram of the values in the current column `col`. The `kde=False` argument specifies that we don't want a density curve overlaid on top of the histogram.
# - `plt.title(f"Distribution of {col}")`: This method sets the title of the current subplot to "Distribution of [column name]".
# - `plt.show()`: This method displays the current subplot. This is necessary because we are showing multiple plots in the same cell, and we need to explicitly tell Matplotlib to display each one before moving on to the next iteration of the loop.

# The resulting output will show a set of histograms, one for each numeric column in `df`, that visualize the distribution of values for each column. The y-axis of each histogram represents the count of occurrences of each value in that column, while the x-axis shows the range of values.

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# execute the info() function
df.info()

# This code prints a summary of the DataFrame `df` using the `info()` method. Here is what each part of the code does:

# - `df.info()`: This method provides a concise summary of the DataFrame `df`. This includes the number of non-null values, column data types, and memory usage of the DataFrame. The resulting output lists the number of rows, the number of columns, the column names, the number of non-null values for each column, and the data type of each column.

# The information printed by `df.info()` is useful for understanding the structure of the DataFrame, including the number of missing values and the data types of each column. This information can help you better prepare your data for analysis by identifying potential issues or inconsistencies in the data.

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for col in df.columns:
   plt.plot(df[col], label=col)
   plt.xlabel('Time')
   plt.ylabel('Temperature (Celsius)')
   plt.title('Temperature over Time')

# This code creates a line plot for each column in the DataFrame `df`. Here is what each part of the code does:

# - `for col in df.columns:`: This `for` loop iterates over each column in the DataFrame `df`.
# - `plt.plot(df[col], label=col)`: This method creates a line plot of the values in the current column `col`. The `label=col` argument assigns the name of the current column as the label for the line. Multiple lines are plotted in the same figure to allow for easy comparison between different columns.
# - `plt.xlabel('Time')`: This method sets the label for the x-axis to "Time".
# - `plt.ylabel('Temperature (Celsius)')`: This method sets the label for the y-axis to "Temperature (Celsius)".
# - `plt.title('Temperature over Time')`: This method sets the title for the plot to "Temperature over Time".

# The resulting output will display a set of line plots, one for each column in `df`. Each plot shows the values in the corresponding column over time. The x-axis is labeled as "Time" and the y-axis is labeled as "Temperature (Celsius)". The title of each plot is "Temperature over Time". This type of visualization is useful for identifying trends and patterns in the data over time.

df.plot(kind='scatter', x='meantempm_1', y='maxtempm_1')
plt.xlabel('Mean Temperature (C)')
plt.ylabel('Max Temperature (C)')
plt.title('Scatter Plot of Mean and Max Temperatures')
plt.show()

# This code creates a scatter plot visualizing the relationship between two columns in the DataFrame `df`. Here is what each part of the code does:

# - `df.plot(kind='scatter', x='meantempm_1', y='maxtempm_1')`: This method creates a scatter plot of the values in the columns 'meantempm_1' and 'maxtempm_1'. The `kind='scatter'` argument specifies that we want a scatter plot, and the `x='meantempm_1'` and `y='maxtempm_1'` arguments specify which columns in `df` should be used for the x and y axes, respectively.
# - `plt.xlabel('Mean Temperature (C)')`: This method sets the label for the x-axis to "Mean Temperature (C)".
# - `plt.ylabel('Max Temperature (C)')`: This method sets the label for the y-axis to "Max Temperature (C)".
# - `plt.title('Scatter Plot of Mean and Max Temperatures')`: This method sets the title for the plot to "Scatter Plot of Mean and Max Temperatures".
# - `plt.show()`: This method displays the plot.

# The resulting output is a scatter plot with 'meantempm_1' on the x-axis and 'maxtempm_1' on the y-axis. Each point in the plot represents a single row in `df`, with the x-coordinate corresponding to the value in 'meantempm_1' and the y-coordinate corresponding to the value in 'maxtempm_1'. The plot can be used to identify any correlation between the two variables - for example, if points tend to cluster in a diagonal line from the lower left to the upper right, this would indicate a positive correlation between the two columns.

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# First drop the maxtempm and mintempm from the dataframe
df = df.drop(['mintempm', 'maxtempm'], axis=1)

# This code drops the columns 'mintempm' and 'maxtempm' from the DataFrame `df`. Here is what each part of the code does:

# - `df.drop(['mintempm', 'maxtempm'], axis=1)`: This method drops the columns specified in the list passed as the first argument ('mintempm' and 'maxtempm') from the DataFrame `df`. The `axis=1` argument specifies that the columns should be dropped (as opposed to rows).
# - `df =`: This assigns the resulting DataFrame with the columns 'mintempm' and 'maxtempm' dropped back to the variable `df`.

# The resulting DataFrame `df` will no longer contain the 'mintempm' and 'maxtempm' columns. This can be useful if these columns are not needed for analysis or visualization, or if they are redundant with other columns in the DataFrame. By removing unnecessary columns, the DataFrame can be reduced in size and made easier to work with.

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# X will be a pandas dataframe of all columns except meantempm
X = df[[col for col in df.columns if col != 'meantempm']]

# This code creates a new DataFrame `X` that includes all columns from the DataFrame `df` except for the column 'meantempm'. Here is what each part of the code does:

# - `[col for col in df.columns if col != 'meantempm']`: This is a list comprehension that creates a list of all column names in the DataFrame `df` except for 'meantempm'. The condition `if col != 'meantempm'` ensures that only columns other than 'meantempm' are included in the list.
# - `df[[col for col in df.columns if col != 'meantempm']]`: This selects all columns from `df` whose names are included in the list created by the list comprehension. The resulting DataFrame `X` contains all columns of `df` except for 'meantempm'.

# The resulting DataFrame `X` can be used for various data analysis tasks, such as building a predictive model or conducting statistical analysis. By removing the target variable 'meantempm' from the DataFrame, we can use the remaining columns as features to predict 'meantempm' or to conduct exploratory data analysis on the other variables in the dataset.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# y will be a pandas series of the meantempm
y = df['meantempm']

# This code creates a new pandas Series `y` that contains the values from the column 'meantempm' in the DataFrame `df`.  

# Here is what each part of the code does:

# - `df['meantempm']`: This selects the column 'meantempm' from the DataFrame `df`. The resulting output is a pandas Series containing the values from the 'meantempm' column.
# - `y = df['meantempm']`: This assigns the output of `df['meantempm']` to a new variable `y`, creating a separate pandas Series that contains only the values from the 'meantempm' column.

# The resulting Series `y` represents the target variable that we want to predict or analyze in our data analysis tasks. This could involve building a predictive model to estimate future 'meantempm' values or conducting statistical analysis to identify relationships between 'meantempm' and other variables in the dataset. Having the target variable separated from the other predictor variables in separate DataFrame and Series allows us to manipulate and analyze them separately based on our analysis requirements.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

# This code uses the `train_test_split` method from the `sklearn.model_selection` module to split our dataset into training and temporary sets. 

# Here is what each part of the code does:

# - `train_test_split(X, y, test_size=0.2, random_state=23)`: This method generates four sets of data, `X_train`, `X_tmp`, `y_train`, and `y_tmp`, by randomly splitting the data in `X` and `y`. The `test_size` argument specifies the proportion of the data to be included in the temporary set (in this case, 20% of the data), and `random_state` provides a seed value to ensure that the same random splits are generated each time the code is run.
# - `X_train`: This is a pandas DataFrame containing the predictor variables in the training set.
# - `y_train`: This is a pandas Series containing the target variable in the training set.
# - `X_tmp`: This is a pandas DataFrame containing the predictor variables in the temporary set.
# - `y_tmp`: This is a pandas Series containing the target variable in the temporary set.

# The purpose of splitting data into training and temporary sets is to prevent overfitting or underfitting of the model when using the complete data set for model validation. The training set is used to train the model, while the temporary set is used to validate the trained model by observing how well the model performs on new and unseen data. This approach helps to ensure that the model can generalize and make accurate predictions on new data.

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

# This code creates a list of feature columns for our dataset that will be used as input to a Tensorflow model. 

# Here is what each part of the code does:

# - `feature_cols`: This is an empty list that will hold our feature columns.
# - `for col in X.columns`: This iterates through each column in the DataFrame `X`.
# - `tf.feature_column.numeric_column(col)`: This creates a new numeric feature column of type `numeric_column` for each column in `X`. The `col` argument specifies the name of the column in `X` that the feature column is created for.

# The resulting list `feature_cols` contains a numeric feature column for each column in `X`. These feature columns can be used to define the input layer of a Tensorflow model which takes in the predictor variables and returns predicted values of the target variable.

# Numeric feature columns represent continuous data and are used to represent input features in a machine learning model that expects real-valued inputs. Numeric feature columns can be used to represent data that can take on any numerical value, such as temperatures, heights, or weights. In this case, we are using numeric feature columns to represent each predictor variable in the dataset.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[50, 50],
                                      model_dir='tf_wx_model')

# This code creates a deep neural network (DNN) regressor model using the `tf.estimator` API from the Tensorflow library. 

# Here is what each part of the code does:

# - `tf.estimator.DNNRegressor()`: This is the constructor method for creating a DNN regressor model with the `tf.estimator` API.
# - `feature_columns=feature_cols`: This argument specifies the feature columns to be used as input to the model. 
# - `hidden_units=[50, 50]`: This argument specifies the architecture of the neural network by defining the number and size of hidden layers in the model. In this case, we are using two hidden layers, each with 50 neurons.
# - `model_dir='tf_wx_model'`: This argument specifies the directory where the trained model is stored on disk.

# The resulting `regressor` object is an instance of the `tf.estimator.DNNRegressor` class, which represents our neural network model. The model is designed to take in the feature columns created earlier as input and generate predictions for the target variable, `meantempm`. The `hidden_units` argument determines the number of neurons in each hidden layer of the model, while the `model_dir` argument specifies the directory where the trained model is stored. 

# Overall, this code represents the creation of a DNN regressor model using Tensorflow, which can be trained using the training data and then used to make predictions on new, unseen data

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=260): # 260 is used as we have approx 570 dataset for training
    return tf.compat.v1.estimator.inputs.pandas_input_fn (x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)

# This code defines a custom input function for our Tensorflow model. The input function is used to provide our model with data during training and evaluation. 

# Here is what each part of the code does:

# - `wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=260)`: This is a user-defined function that takes in the following arguments:
#     - `X`: This is a pandas DataFrame containing the predictor variables (features) for the model.
#     - `y=None`: This is a pandas Series containing the target variable for the model (optional).
#     - `num_epochs=None`: This is an integer that specifies the number of times to iterate over the data (optional). If set to `None`, the function will iterate indefinitely until stopped.
#     - `shuffle=True`: This is a boolean that specifies whether or not to shuffle the data (optional). 
#     - `batch_size=260`: This is an integer that specifies the number of examples to include in each batch of data (optional). 

# - `tf.compat.v1.estimator.inputs.pandas_input_fn()`: This is a method from Tensorflow which returns an input function that feeds data into an Estimator by returning a generator object that can be used as input to `train`, `evaluate`, and `predict` methods of the model. 

# The `wx_input_fn` function is essentially a wrapper around the `pandas_input_fn` method, and it uses the arguments passed to it to configure the input function. Specifically, the input function `pandas_input_fn` returns a generator object that yields batches of data from our `X` and `y` datasets. The `batch_size` argument specifies the number of samples to include in each batch. The `shuffle` argument specifies whether we want to randomize the order of the dataset during training or evaluation. The `num_epochs` argument specifies the number of times to iterate over the entire dataset. If unspecified, the generator will continue to produce batches indefinitely until the `num_epochs` limit is reached. 

# Overall, `wx_input_fn` is a convenience function that returns the `pandas_input_fn` generator object with pre-defined configuration values for the arguments that create a suitable input function for training and evaluating our DNN regressor model.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

evaluations = []
STEPS = 260
for i in range(100):
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluation = regressor.evaluate(input_fn=wx_input_fn(X_val, y_val,
                                                         num_epochs=1,
                                                         shuffle=False),
                                    steps=1)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))
    

# This code trains and evaluates our DNN regressor model.

# Here is what each part of the code does:

# - `evaluations = []`: This creates an empty list to store evaluation metrics for each iteration of training.
# - `STEPS=260`: This sets the number of training steps for each iteration.
# - `for i in range(100)`: This loops through 100 iterations of the training process. Each iteration consists of a training and evaluation step.
# - `regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)`: This trains the DNN regressor model using data from the training set. The `input_fn` argument specifies the custom input function that we defined earlier. The `steps` argument specifies the number of training steps to run.
# - `regressor.evaluate(input_fn=wx_input_fn(X_val, y_val, num_epochs=1, shuffle=False), steps=1)`: This evaluates the performance of the DNN regressor model using data from the validation set. The `input_fn` argument specifies the custom input function that we defined earlier. The `num_epochs` argument is set to 1 to ensure that the generator only iterates over the validation set once. The `shuffle` argument is set to `False` to ensure that the examples are evaluated in their original order. The `steps` argument specifies the number of batches to evaluate. Here, we are evaluating on only one batch of data.

# - `evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,y_val,num_epochs=1,shuffle=False)))`: This line of code evaluates the model on the validation set again and appends the evaluation metrics to the `evaluations` list.

# In summary, this code trains and evaluates the DNN regressor model on the training and validation data, respectively, for 100 iterations. During each iteration, the model is trained using the `train` method and evaluated using the `evaluate` method. The evaluation metrics are stored in the `evaluations` list for future analysis.

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

evaluations[0]

# This code retrieves the evaluation metrics for the first iteration of the training and evaluation loop.

# - `evaluations[0]` retrieves the first item from the `evaluations` list, which contains the evaluation metrics from the first iteration of the training and evaluation process.
# - The evaluation metrics are stored as a dictionary, where the keys are the names of the metrics and the values are the evaluated metric values. For example, if we wanted to retrieve the evaluation metric for the mean absolute error (MAE), we could use `evaluations[0]['average_loss']`, since the MAE is computed as the average loss across all examples in the validation set for the first iteration. 

# Overall, this code retrieves the evaluation metrics for the first iteration of the training and evaluation process, which can be used to measure the initial performance of the model on the validation set.

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)
plt.xlabel('Training steps (Epochs = steps / 2)')
plt.ylabel('Loss (SSE)')
plt.show()

# This code generates a scatter plot of the SSE loss values for the DNN regressor model during the training process.

# Here is what each part of the code does:

# - `plt.rcParams['figure.figsize'] = [14, 10]`: This sets the size of the figure in inches. A size of 14 inches wide by 10 inches tall is defined.
# - `loss_values = [ev['loss'] for ev in evaluations]`: This creates a list of SSE loss values for each evaluation in the `evaluations` list. SSE loss, or sum of squared errors, is a common metric used to evaluate regression models.
# - `training_steps = [ev['global_step'] for ev in evaluations]`: This creates a list of global step values for each evaluation in the `evaluations` list. Global step is a counter that tracks the number of training steps completed during training.
# - `plt.scatter(x=training_steps, y=loss_values)`: This creates a scatter plot of the SSE loss values versus the global step values. Each point on the plot represents an evaluation of the model at a certain number of training steps. The `x` and `y` arguments specify the data to be plotted.
# - `plt.xlabel('Training steps (Epochs = steps / 2)')`: This adds a label to the x-axis of the plot indicating that the values represented by `x` are training steps with half the number of epochs as `steps`.
# - `plt.ylabel('Loss (SSE)')`: This adds a label to the y-axis of the plot indicating that the values represented by `y` are SSE loss values.
# - `plt.show()`: This displays the scatter plot in a new window.

# Overall, this code creates a scatter plot of the SSE loss values during the training process of the DNN regressor model. The plot provides an indication of how the model's performance improves over time as the number of training steps increase.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])


print("The Explained Variance: %.2f" % explained_variance_score(
                                            y_test, predictions))  
print("The Mean Absolute Error: %.2f celcius" % mean_absolute_error(
                                            y_test, predictions))  
print("The Median Absolute Error: %.2f celcius" % median_absolute_error(
                                            y_test, predictions))
print("Test Accuracy : {} %".format(np.mean(np.abs(predictions-y_test))*100))
mape = np.mean(np.abs(predictions - y_test) / np.abs(y_test)) * 100
print("Test MAPE: {:.2f}%".format(mape))

# This code evaluates the performance of the DNN regressor model on the test dataset.

# Here is what each part of the code does:

# - `pred = regressor.predict(input_fn=wx_input_fn(X_test, num_epochs=1, shuffle=False))`: This generates predictions from the trained DNN regressor model for the test dataset. The `input_fn` argument specifies the custom input function that we defined earlier. The `num_epochs` argument is set to 1 to ensure that the generator only iterates over the test set once. The `shuffle` argument is set to `False` to ensure that the examples are evaluated in their original order.

# - `predictions = np.array([p['predictions'][0] for p in pred])`: This converts the predicted values from the `regressor.predict` generator object into a numpy array. The predicted values are stored as a single value in a dictionary with the key `'predictions'`. We take only the first value from this dictionary and store it in a numpy array.

# - `explained_variance_score(y_test, predictions)`: This computes the explained variance score between the true values `y_test` and the predicted values `predictions`. The explained variance score is a measure of how well the model fits the data. A score of 1.0 indicates a perfect fit, while a score of 0.0 indicates that the model does no better than predicting the mean of the target variable.

# - `mean_absolute_error(y_test, predictions)`: This computes the mean absolute error (MAE) between the true values `y_test` and the predicted values `predictions`. The MAE is a measure of the average magnitude of the errors in the predictions, with a smaller value indicating better performance. 

# - `median_absolute_error(y_test, predictions)`: This computes the median absolute error between the true values `y_test` and the predicted values `predictions`. The median absolute error is another measure of the average magnitude of the errors in the predictions, but is more robust to outliers than the mean absolute error.

# - `np.mean(np.abs(predictions-y_test))*100`: This computes the mean absolute percentage error (MAPE) between the true values `y_test` and the predicted values `predictions`. The MAPE is a measure of the average magnitude of the errors as a percentage of the true values. A smaller value indicates better performance. 

# - `mape = np.mean(np.abs(predictions - y_test) / np.abs(y_test)) * 100`: This calculates the MAPE value for the predictions and stores it in `mape` variable.

# Overall, this code evaluates the performance of the DNN regressor model on the test dataset by generating predictions and computing several evaluation metrics, including the explained variance score, MAE, median absolute error, mean absolute percentage error (MAPE), and test accuracy. These metrics provide a measure of how well the model generalizes to new, unseen data.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

compare = pd.DataFrame({'actual': y_test, 'predicted': predictions})
compare

# This code creates a Pandas DataFrame that compares the actual and predicted values of the target variable for the test dataset.

# Here is what each part of the code does:

# - `pd.DataFrame()`: This creates a new Pandas DataFrame object.

# - `{'actual': y_test, 'predicted': predictions}`: This dictionary specifies the columns of the new DataFrame. The `'actual'` column contains the true target variable values from the test dataset (`y_test`), while the `'predicted'` column contains the predicted target variable values computed by the DNN regressor model (`predictions`).

# - `compare`: This assigns the new DataFrame object to the variable `compare`. 

# Overall, this code creates a new DataFrame that can be used to compare the actual and predicted values of the target variable for the test dataset. The DataFrame can be further analyzed or visualized to gain insights into the model's performance.

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for i in range(68):
    print('original value: ', y_test[i])
    print('predicted value: ', predictions[i])

# This code prints the actual and predicted values of the target variable for the first 68 examples in the test dataset. 

# Here is what each part of the code does:

# - `for i in range(68)`: This sets up a loop that iterates from `i = 0` to `i = 67`, which corresponds to the first 68 examples in the test dataset.

# - `print('original value: ', y_test[i])`: This prints the true value of the target variable for the `i`'th example in the test dataset, with a description of `'original value: '` added to the beginning of the output.

# - `print('predicted value: ', predictions[i])`: This prints the predicted value of the target variable for the `i`'th example in the test dataset, with a description of `'predicted value: '` added to the beginning of the output.

# Overall, this code provides a quick way to visually compare the actual and predicted values of the target variable for some examples in the test dataset. The loop iterates over the first 68 examples and prints the corresponding true and predicted values for each example. This can help to give a sense of how well the model is performing on the test dataset, but is not a comprehensive evaluation of the model's performance.    

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, predictions)))

# This code computes the R-squared score, which is a measure of how well the regression model fits the data. 

# Here is what each part of the code does:

# - `r2_score(y_test, predictions)`: This computes the R-squared score between the true values `y_test` and the predicted values `predictions`. The R-squared score ranges from -1.0 to 1.0, where a score of 1.0 indicates a perfect fit, a score of 0.0 indicates that the model does no better than predicting the mean of the target variable, and a score less than 0.0 indicates a worse fit than predicting the mean.

# - `print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, predictions))) `: This prints the R-squared score on the test set with a description of `"The R2 score on the Test set is:"`. The `{:0.3f}` format specifier ensures that the R-squared score is printed to three decimal places.

# Overall, this code provides an additional evaluation metric of the performance of the regression model. The R-squared score provides an indication of how much of the variance in the target variable is explained by the model, with a higher score indicating a better fit to the data.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('X_train : ',X_train.shape)
print('y_train : ',y_train.shape)
print('X_test : ',X_test.shape)
print('y_test : ',y_test.shape)

# This code splits the dataset into training and testing sets using scikit-learn's `train_test_split` function.

# Here is what each part of the code does:

# - `X`: This is the array of input features for the regression model.

# - `y`: This is the array of target variable values for the regression model.

# - `train_test_split(X, y, test_size=0.2)`: This function splits the input features array `X` and the target variable array `y` into two subsets: a training set and a testing set. The `test_size` argument specifies the proportion of the data to be included in the test set (in this case, 20% of the data).

# - `X_train`: This is the array of input features in the training set, returned by `train_test_split`.

# - `y_train`: This is the array of target variable values in the training set, returned by `train_test_split`.

# - `X_test`: This is the array of input features in the testing set, returned by `train_test_split`.

# - `y_test`: This is the array of target variable values in the testing set, returned by `train_test_split`.

# - `print('X_train : ',X_train.shape)`: This prints the shape of the `X_train` array, which is the number of examples in the training set by the number of input features.

# - `print('y_train : ',y_train.shape)`: This prints the shape of the `y_train` array, which is the number of examples in the training set by the number of target variables.

# - `print('X_test : ',X_test.shape)`: This prints the shape of the `X_test` array, which is the number of examples in the test set by the number of input features.

# - `print('y_test : ',y_test.shape)`: This prints the shape of the `y_test` array, which is the number of examples in the test set by the number of target variables.

# Overall, this code splits the dataset into two subsets: a training set and a testing set. The training set is used to fit the regression model, while the testing set is used to evaluate the model's performance on new, unseen data. The shapes of the resulting arrays are printed to verify that the split was performed correctly.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# This code performs feature scaling on the input feature arrays of the regression model, using scikit-learn's `preprocessing` module.

# Here is what each part of the code does:

# - `preprocessing.scale()`: This function standardizes the input feature array by subtracting the mean and scaling to unit variance. It is a common preprocessing step in machine learning to ensure that all features have a similar scale and are centered around zero.

# - `X_train = preprocessing.scale(X_train)`: This applies feature scaling to the `X_train` array, so that each input feature has zero mean and unit variance within the training set.

# - `X_test = preprocessing.scale(X_test)`: This applies feature scaling to the `X_test` array, using the same scaling parameters as the `X_train` array. It is important to use the same scaling parameters for both the training and testing sets to ensure that the model is evaluated on data that is consistent with what it was trained on.

# Overall, this code standardizes the input feature arrays to improve the performance of the regression model. Standardization makes it easier for the model to learn the relevant patterns in the data, especially when the input features have different scales. By scaling the features, we ensure that all features have the same scale and are centered around zero, which makes them more comparable and helps the model to better generalize to new, unseen data.

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#applying the neurons in the layers with weights and biases to work on the algorithm....linear regression is applied... y =mx_c.
model = Sequential()
model.add(Dense(13, input_shape=(36,), activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1,))
model.compile(Adam(lr=0.003), 'mean_squared_error')

# This code creates a feedforward neural network regression model using Keras, with the following architecture:

# - The `Sequential()` function initializes an empty model.

# - The `Dense()` function adds a layer of fully connected neurons to the model. The `input_shape` argument specifies the size of the input to the first layer, which in this case is a vector of length 36 (the number of input features). The `activation` argument specifies the activation function used by the neurons in the layer, which in this case is the rectified linear unit (ReLU).

# - There are five hidden layers, each with 13 neurons and a ReLU activation function.

# - The output layer has one neuron, which outputs the predicted target variable value.

# - The `model.compile()` function compiles the model with an optimizer and a loss function. The optimizer, `Adam(lr=0.003)`, is a stochastic gradient descent (SGD) algorithm that adjusts the weights and biases of the neural network during training to minimize the loss function. The learning rate (`lr=0.003`) specifies the step size of the optimizer at each iteration. The loss function, `'mean_squared_error'`, measures the difference between the true target variable values and the predicted values produced by the model, using the mean square error (MSE) as the error metric.

# Overall, this code creates a neural network regression model with five hidden layers and an output layer, using the rectified linear unit (ReLU) activation function. The model is compiled with an optimizer and a loss function to minimize the mean squared error (MSE) during training.

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Runs model for 2000 iterations and assigns this to 'history'
model.summary()

# `model.compile()` initializes the model's optimizer, loss function, and evaluation metrics. The `compile` method compiles the Keras model and returns a tensor graph to represent the model internally. The `metrics` argument specifies the list of evaluation metrics that the model should use to measure the performance. In this case, the metric is `'mean_squared_error'`.

# `model.summary()` provides a summary of the model's architecture, including the number of layers, the number of parameters and output shape of each layer. It summarizes the output shape of each layer in the model, including the number of trainable parameters in each layer. This is useful for understanding the architecture of the model and for troubleshooting any errors or issues that may arise during training.

# Overall, `model.compile()` defines the loss function and optimizer used to train the model and `model.summary()` gives information about the structure of the model, such as the number of layers, the number of parameters, and the output shape of each layer.

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

history = model.fit(X_train, y_train, epochs = 100,
                    validation_split = 0.2, verbose = 1)

# `model.fit()` trains the neural network model on the input and target values. The `fit()` method trains the model by minimizing the loss function using the provided optimizer on the training data.

# Here is what each part of the code does:

# - `X_train` and `y_train` are the training input features and target variable values, respectively, that the model will be trained on.

# - `epochs` is the number of times the model will iterate through the entire training dataset. In this case, the model will iterate through the dataset 100 times.

# - `validation_split` is the fraction of training data to be used as validation data. In this case, 20% of the training data is used as validation data.

# - `verbose` sets the level of detail displayed during training. When `verbose = 1`, a progress bar is displayed during training.

# - `history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)` stores the training history of the model in the `history` variable.

# Overall, `model.fit()` trains the neural network model on the input and target values for a specified number of epochs with a validation split. The training history is saved in the `history` variable for later analysis.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
hist.tail()

# `pd.DataFrame(history.history)` creates a Pandas DataFrame object using the history dictionary returned by the `fit()` method of the model. This DataFrame object contains the training and validation loss values for each epoch of the training process.

# - `hist['epoch'] = history.epoch` adds an additional column to the `hist` DataFrame containing the epoch number for each row.

# - `hist.tail()` returns the last five rows of the `hist` DataFrame.

# Overall, this code creates a Pandas DataFrame object containing loss values for each epoch and the corresponding epoch number. Later, this DataFrame object can be used for data analysis and visualization of the training history of the model.

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

# `model.predict()` returns the predicted target variable values using the trained neural network model. The `predict()` method takes the input features as input and produces the predicted target variable values as output.

# - `y_train_pred` and `y_test_pred` are the predicted target variable values for the training and test data, respectively, obtained by using the `predict()` method of the model on the input feature data `X_train` and `X_test`.

# - `r2_score()` computes the R-squared (coefficient of determination) regression score function. R-squared is a statistical measure that represents the goodness of fit of a linear regression model. It measures the proportion of the variation in the dependent variable (y) that is predictable from the independent variable (X). 

# - `print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))` prints the R-squared score for the training data.

# - `print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred))) ` prints the R-squared score for the test data.

# Overall, this code generates predicted target variable values for both the training and test dataset using the neural network model. It then computes R-squared scores for the predicted and actual target variable values for both the training and test dataset. The R-squared score is a measure of how well the neural network model fits the dataset.

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.title('model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_values,'bo',label='training loss')

plt.plot(val_loss_values,'r',label='validation loss ')
plt.legend(['train', 'validation'], loc='upper right')

# This code generates a line plot of the training and validation loss values for each epoch during the training process of the neural network model.

# - `history_dict=history.history` saves the history dictionary returned by the `fit()` method of the model to the `history_dict` variable.

# - `loss_values = history_dict['loss']` and `val_loss_values=history_dict['val_loss']` extracts the training and validation loss values from the `history_dict` dictionary.

# - `plt.title('model Loss')` sets the title of the plot to 'Model Loss'.

# - `plt.ylabel('loss')` sets the y-axis label to 'loss'.

# - `plt.xlabel('epoch')` sets the x-axis label to 'epoch'.

# - `plt.plot(loss_values,'bo',label='training loss')` creates a scatter plot of the training loss values from the `loss_values` array with blue dots ('bo') and labeled 'Training Loss'.

# - `plt.plot(val_loss_values,'r',label='validation loss ')` creates a line plot of the validation loss values from the `val_loss_values` array with a red line ('r') and labeled 'Validation Loss'.

# - `plt.legend(['train', 'validation'], loc='upper right')` adds a legend to the plot, showing the labels for 'train' and 'validation' with upper-right positioning.

# Overall, this code generates a data visualization of the training and validation loss values for each epoch of the training process for the neural network model. This visualization is useful for evaluating the performance of the model and identifying overfitting or underfitting issues.






























































