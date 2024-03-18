import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import keras.utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import keras.optimizers as ko
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

print(' --- FIRST MODEL - LOGISTIC REGRESSION --- ')

# Set random seed
np.random.seed(10)
keras.utils.set_random_seed(10)

# Replace missing values and Split the data
data = np.load('/Users/valentinlupu/Downloads/data.npz')
X = data['features']
y = data['labels']
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=1)

imp = SimpleImputer(strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 3, random_state=1)

# Alternative values for hyperparameters
C_values = [0.01, 1, 100]
penalties = ['l1', 'l2']
solvers = ['liblinear']
markers = ['o']

# GridSearch to find the best hyperparameters
start_time = time.time()

param_grid = {'C': C_values, 'solver': solvers, 'penalty': penalties}

grid_search = GridSearchCV(LogisticRegression(max_iter=100000), param_grid, cv=kfold)
grid_search.fit(X_train, y_train)
print("Training accuracy: {:.2f}%".format(grid_search.score(X_train, y_train) * 100))
print("Validation accuracy: {:.2f}%".format(grid_search.score(X_val, y_val) * 100))
print("Best estimator:\n{}".format(grid_search.best_estimator_))

# Final training
lr_l1 = grid_search.best_estimator_.fit(X_train, y_train)
training_time = (time.time() - start_time)

# Print final accuracy
print(f"""
Final Accuracy: {lr_l1.score(X_test, y_test)* 100:.2f}%
Best hyperparameters: {grid_search.best_params_}
Training_time: {training_time}
""")

# Plot Confusion Matrix
y_pred = lr_l1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)
ax.set_title('Logistic Regression')
plt.show()


print(' --- SECOND MODEL - EARLYSTOPPING NEURAL NETWORK --- ')

# Replace missing values and Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=1)

imp = SimpleImputer(strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

# Scale and Encode the labels
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Apply Principal Component Analysis
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Split training data for cross-validation training
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1 / 3, random_state=1)

# Add hyperparameters for testing
batch_sizes = [32]
acts = ['tanh']
n_units = [32]
layers = [1, 3, 5, 7]
l_rates = [0.001]
opts = [ko.legacy.Adam, ko.legacy.RMSprop, ko.legacy.SGD]

# Test hyperparameters
start_time = time.time()

best_accuracy = 0

for unit in n_units:
    for layer in layers:
        for lr in l_rates:
            for opt in opts:
                for act in acts:

                    optimizer = opt(learning_rate=lr)
                    model = Sequential()
                    model.add(Dense(32, activation=act, input_dim=10))
                    for n in range(layer):
                        model.add(Dense(32, activation=act))
                    model.add(Dense(4, activation='softmax'))
                    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                    monitor_val_acc = EarlyStopping(monitor='val_accuracy', patience=10)
                    model_checkpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only=True)
                    model.fit(X_train, y_train, epochs=1000, callbacks=[monitor_val_acc, model_checkpoint],
                              validation_data=(X_val, y_val), batch_size=32, verbose=0)

                    training_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
                    model_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
                    print(round(training_accuracy, 4), 'train_accuracy', round(model_accuracy, 4), 'val_accuracy', unit, 'units', act, 'activation',  lr, 'l_rate', opt.__name__, 'optimizer', layer+1, 'layers')

                    if model_accuracy > best_accuracy:
                        best_accuracy = model_accuracy
                        best_unit = unit
                        best_layer = layer
                        best_lr = lr
                        best_opt = opt
                        best_act = act

# Final training
optimizer = best_opt(learning_rate=best_lr)
model = Sequential()
model.add(Dense(best_unit, activation=best_act, input_dim=10))
for n in range(best_layer):
    model.add(Dense(best_unit, activation=best_act))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

monitor_val_acc = EarlyStopping(monitor='val_accuracy', patience=10)
model_checkpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only=True)
history = model.fit(X_train, y_train, epochs=1000, callbacks=[monitor_val_acc, model_checkpoint],
                    validation_data=(X_test, y_test), batch_size=32, verbose=0)

model_loss, model_accuracy = model.evaluate(X_test, y_test, verbose=1)

training_time = (time.time() - start_time)

# Print final accuracy and best hyperparameters
print(f'\nTest Accuracy: {model_accuracy * 100:.2f}%')
print(f'''Final model: 
{best_unit} units
{best_layer + 1} layers
{best_lr} learning rate
{best_opt.__name__} optimizer
{best_act} activation 
''')
print("Training time: %s seconds" % training_time)

# Plot training process
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Neural Network Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

# Plot Confusion Matrix
predictions = model.predict(X_test)
predictions = predictions.argmax(axis=1)
y_test_final = y_test.argmax(axis=1)

cm = confusion_matrix(y_test_final, predictions)
disp = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)
ax.set_title('Neural Network')
plt.show()