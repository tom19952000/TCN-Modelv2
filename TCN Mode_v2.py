import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import time
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tcn import TCN, tcn_full_summary
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers import LeakyReLU


######################################################################################################################
# Start measuring time
start_time = time.time()

#Raw data source
df = pd.read_csv(r'C:\Users\user\Desktop\HKU\Year 1 - Semester 1\COMP7409A - Machine learning in trading and finance\Group Project\train.csv')

###########################################################################################################################################

#Using Foward Fill to fill the NaN value
df.ffill(inplace=True)

#For those initial value is NaN (unable to use Forward Fill), replace it as 0 (n=30, so won't impact much)
df['far_price'].fillna(1e-6, inplace=True)
df['near_price'].fillna(1e-6, inplace=True)

#Explanation
    #len(result_df) = 5,291,000
    # (# of days) * (# of time buckets in 1 day) * (# of stock_id)
    # 481 (from 0 to 480) * 55 (from 0 to 540) * 200 (from 0 to 199)
    # 481*55*200
    # = 5,291,000

###########################################################################################################################################

#Data Preparation

# Split data
test_df = df[-33000:]
remaining_df = df[:-33000]
train_df, cv_df = train_test_split(remaining_df, test_size=0.40)

# Prepare X and y datasets
columns_to_drop = ['stock_id', 'date_id', 'seconds_in_bucket', 'time_id', 'row_id', 'target']
x_train = train_df.drop(columns=columns_to_drop)
y_train = train_df['target'].values
x_cv = cv_df.drop(columns=columns_to_drop)
y_cv = cv_df['target'].values
x_test = test_df.drop(columns=columns_to_drop)
y_test = test_df['target'].values


###########################################################################################################################################

# TCN Model parameters
sequence_length = 55
# Assuming you want to keep the same number of features
num_features = x_train.shape[1]

total_samples = x_train.shape[0]
num_sequences = total_samples // sequence_length

# Ensuring the number of samples in x_train, x_cv and x_test are divisible by sequence_length
def trim_to_sequence_length(data, sequence_length):
    remainder = data.shape[0] % sequence_length
    if remainder != 0:
        trim_size = data.shape[0] - remainder
        data = data[:trim_size]
    return data

x_train = trim_to_sequence_length(x_train, sequence_length)
y_train = trim_to_sequence_length(y_train, sequence_length)
x_cv = trim_to_sequence_length(x_cv, sequence_length)
y_cv = trim_to_sequence_length(y_cv, sequence_length)
x_test = trim_to_sequence_length(x_test, sequence_length)
y_test = trim_to_sequence_length(y_test, sequence_length)

# Reshape the data
x_train = x_train.values.reshape(-1, sequence_length, num_features)
y_train = y_train.reshape(-1, sequence_length, 1)
x_cv = x_cv.values.reshape(-1, sequence_length, num_features)
y_cv = y_cv.reshape(-1, sequence_length, 1)
x_test = x_test.values.reshape(-1, sequence_length, num_features)
y_test = y_test.reshape(-1, sequence_length, 1)



################################################################################################

# Define the TCN model architecture
model = keras.models.Sequential()
model.add(TCN(input_shape=(sequence_length, num_features), return_sequences=True))  # The TCN layers are added here
model.add(TCN(return_sequences=True))
model.add(keras.layers.Dense(1, activation='tanh'))



# Compile the model
#optimizer = Adam(learning_rate=0.001)
optimizer = Adagrad(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
tcn_full_summary(model, expand_residual_blocks=True)
###########################################################################################################

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=200, validation_data=(x_cv, y_cv))

# Make predictions on the test set
y_pred_sequences = model.predict(x_test)

# Flatten the predictions to obtain individual predictions
y_pred = y_pred_sequences.flatten()

# testing
#print("y_test shape:", y_test.shape)
#print("y_pred_sequences shape:", y_pred_sequences.shape)
#print("y_pred shape:", y_pred.shape)

# Calculate evaluation metrics (e.g., Mean Squared Error)
mse_test = mean_squared_error(y_test.flatten(), y_pred)
print("Mean Squared Error for testing:", mse_test)

# Ensure that the lengths of y_pred and y_test are the same
expected_num_predictions = y_test.flatten().shape[0]
assert len(y_pred) == expected_num_predictions, f"Length Mismatch: {len(y_pred)} vs {expected_num_predictions}"

# Save the relevant predictions to the CSV
predictions_df = pd.DataFrame({'Predicted_Target': y_pred})
predictions_df.index = range(0, expected_num_predictions)  # Set the index for the output
predictions_df.to_csv('predicted_targets_raw.csv')

# Record the end time
end_time = time.time()

# Calculate and print the running time
running_time = end_time - start_time
print("Running Time (seconds):", running_time)
