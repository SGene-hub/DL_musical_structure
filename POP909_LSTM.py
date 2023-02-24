import argparse
import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy




parser = argparse.ArgumentParser(
    description="POP909 LSTM",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--units", default=100, type=int, help="kernel size")
parser.add_argument("--dropout-rate", default=0.2, type=float, help="dropout-rate")
parser.add_argument("--pool-size", default=2, type=int, help="pool size")
parser.add_argument("--pool-stride", default=2, type=int, help="pool stride")
parser.add_argument("--learning-rate", default=1e-4, type=float, help="learning rate")
parser.add_argument("--focal-gamma", default=2.0, type=float, help="focal gamma")
parser.add_argument("--epochs", default=10, type=int, help="epochs")
parser.add_argument("--batch-size", default=100, type=int, help="batch size")
args = parser.parse_args()


tf.random.set_seed(1234)

# Read data
input = np.load('data/processed_input_pad.npy') 
output1 = np.load('data/processed_output_pad_human1.npy')
output2 = np.load('data/processed_output_pad_human2.npy')
mask = np.random.uniform(size=input.shape[0]) < 0.8

# Create train and test set
x_train = np.vstack((input[mask,:], input[mask,:]))
y_train = np.vstack((output1[mask,:], output2[mask,:]))

x_test = np.vstack((input[~mask,:], input[~mask,:]))
y_test = np.vstack((output1[~mask,:], output2[~mask,:]))

N_test = x_test.shape[0]
N_train = x_train.shape[0]

img_shape_full = x_train.shape[1:] # (height, width, n_channels)
output_shape = y_train.shape[1]

# Define model
model = Sequential()


model.add(InputLayer(input_shape=img_shape_full,))

model.add(LSTM(args.units, return_sequences=True, input_shape=img_shape_full))
model.add(Dropout(args.dropout_rate))
model.add(LSTM(args.units, return_sequences=True))
model.add(Dropout(args.dropout_rate))
model.add(LSTM(args.units))
model.add(Dropout(args.dropout_rate))

model.add(Dense(output_shape, activation='sigmoid'))

# Compile model
optimizer = Adam(lr=args.learning_rate)
loss = BinaryFocalCrossentropy(gamma=args.focal_gamma)
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['BinaryAccuracy', 'BinaryIoU'])

# Fit model
model.fit(x=x_train, 
          y=y_train, 
          epochs=args.epochs, 
          batch_size=args.batch_size,
          validation_data=(x_test, y_test))

# Store the result
history = model.history.history

result = model.evaluate(x=x_test,
                        y=y_test)
evaluate_result = {}
for name, value in zip(model.metrics_names, result):
    evaluate_result[name] = value

#TODO: save history and evaluate_result along with args
# Maybe plot history and save fig for early assessment
# Maybe save all evaluation binary iou in one table for easy sorting


hyperparameters = vars(args)
fileroot = '-'.join([f'{name}={value}' for name, value in hyperparameters.items()])

with open(f"results/LSTM-{fileroot}-history", 'wb') as f:
        pickle.dump(history, f)
with open(f"results/LSTM-{fileroot}-evaluate_result", 'wb') as f:
        pickle.dump(evaluate_result, f)