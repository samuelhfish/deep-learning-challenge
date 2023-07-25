# deep-learning-challenge
Deep learning exercise using machine learning and neural networks.


All code exists in the jupyter notebooks in the "notebooks" folder. All HDF5 files are in the "h5_files" folder.

Google colab notebooks were used for this assignment and are accessible here:
https://drive.google.com/drive/folders/1lpEeT7assB-A-LTnWod1GNRYHjdO7wZO?usp=sharing

## OVERVIEW

The goal of this exercise is to create a tool that will help the Alphabet Soup Foundation select applicants for funding with the best chance of sucess in their ventures. We received a CSV with more than 34000 organizations that Alphabet Soup funded over the years.

The information included:

```
EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively
```

Using neural networks, we wanted to use some or all of the first 10 data columns to predict the final column, whether or not the venture was successful.


## RESULTS

### Data Prepoccessing

To increase the efficacy of the model we wanted to get rid of unnessarry variables like the EIN number and the name of the ventue. We also wanted to group excessive variables in the "classification" and "type" columns that only appeared sporadically into an "other" category to hopefully increase the efficacy of our models. Our target variable was the "IS_SUCCESSFUL" column while the remaining columns would be the "features" in our model.

```python

# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df.drop(columns=["EIN","NAME"], inplace=True)

# Determine the number of unique values in each column.
application_df.nunique()

# Look at APPLICATION_TYPE value counts for binning
type_count = application_df["APPLICATION_TYPE"].value_counts()
type_count

# Choose a cutoff value and create a list of application types to be replaced
# use the variable name `application_types_to_replace`
application_types_to_replace = list(type_count[type_count<500].index)

# Replace in dataframe
for app in application_types_to_replace:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app,"Other")

# Check to make sure binning was successful
application_df['APPLICATION_TYPE'].value_counts()

# Look at CLASSIFICATION value counts for binning
class_count = application_df["CLASSIFICATION"].value_counts()
class_count

# Choose a cutoff value and create a list of classifications to be replaced
# use the variable name `classifications_to_replace`
classifications_to_replace = list(class_count[class_count<1000].index)

# Replace in dataframe
for cls in classifications_to_replace:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls,"Other")

# Check to make sure binning was successful
application_df['CLASSIFICATION'].value_counts()

# Convert categorical data to numeric with `pd.get_dummies`
application_df.head()

application_df.dtypes

dummies_df = pd.get_dummies(application_df)
dummies_df.head()

# Split our preprocessed data into our features and target arrays
y = dummies_df.IS_SUCCESSFUL.values
# y = dummies_df.["IS_SUCCESSFUL"].values
X = dummies_df.drop(columns="IS_SUCCESSFUL").values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

### Compiling, Training and Evaluating the Model

After preparing and scaling our data we first started with a model with 2 hidden layers and one ouput layer with the hidden layers having 80 and 30 neurons respectively and the output having 2 for succesful or unsucceful.

```python

# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1=80
hidden_nodes_layer2=30


nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn.fit(X_train,y_train,epochs=100)


# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
```
268/268 - 1s - loss: 1.2683 - accuracy: 0.5895 - 500ms/epoch - 2ms/step
Loss: 1.2683279514312744, Accuracy: 0.5895043611526489
```


Our initial model had a very low accuracy and high loss so we tinkered with by changing variables to test the results. In our first optimization model we dropped two more columns from the initial table and dropped our neuron count to 8 and 3 in the two hidden layers.

```python
# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
application_df.drop(columns=["EIN","NAME","SPECIAL_CONSIDERATIONS","STATUS"], inplace=True)
```

```python
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1=8
hidden_nodes_layer2=3
```

With this we were able to get the loss down to 63% and the accuracy up to almost 70%

```
268/268 - 0s - loss: 0.6329 - accuracy: 0.6941 - 461ms/epoch - 2ms/step
Loss: 0.6328510046005249, Accuracy: 0.6941108107566833
```

For our second optiization model, we again dropped only the "NAME" and "EIN" mode but added a hidden layer for a total of three with 90, 20 and 10 neurons a pieces. This time we tried a hyberbolic tangen function instead of ReLU function. However, our loss remained high and accuracy low with thes variables

```python
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1=90
hidden_nodes_layer2=20
hidden_nodes_layer3=10


nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="tanh")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="tanh"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="tanh"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

```
268/268 - 1s - loss: 0.6994 - accuracy: 0.4980 - 550ms/epoch - 2ms/step
Loss: 0.6993572115898132, Accuracy: 0.4979591965675354
```

For our final try at optimization, we used a a sequential model and kerastuner to predict the best hyperparameter options. This tries different functions and hyperparameters to find the most accurate model.

```python
# Create a method that creates a new Sequential model with hyperparameter options
def create_model(hp):
    nn_model = tf.keras.models.Sequential()

    # Allow kerastuner to decide which activation function to use in hidden layers
    activation = hp.Choice('activation',['relu','tanh','sigmoid'])
    
    # Allow kerastuner to decide number of neurons in first layer
    nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units',
        min_value=1,
        max_value= 30,
        step=5), activation=activation, input_dim=43))

    # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
    for i in range(hp.Int('num_layers', 1, 5)):
        nn_model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
            min_value=1,
            max_value=30,
            step=5),
            activation=activation))
    
    nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Compile the model
    nn_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    return nn_model

# Import the kerastuner library
import keras_tuner as kt

tuner = kt.Hyperband(
    create_model,
    objective="val_accuracy",
    max_epochs=20,
    hyperband_iterations=2)
```

```python
# Run the kerastuner seaarch for best hyperparameters.
tuner.search(X_train_scaled,y_train,epochs=20,validation_data=(X_test_scaled,y_test))
```
```
Trial 56 Complete [00h 00m 56s]
val_accuracy: 0.7216326594352722

Best val_accuracy So Far: 0.7314285635948181
Total elapsed time: 00h 23m 51s
```

```python
# Top 3 model hyperparameters 
top_hyper = tuner.get_best_hyperparameters(3)
for param in top_hyper:
  print(param.values)
```
```

# Top 3 model hyperparameters 
top_hyper = tuner.get_best_hyperparameters(3)
for param in top_hyper:
  print(param.values)
{'activation': 'tanh', 'first_units': 9, 'num_layers': 3, 'units_0': 11, 'units_1': 1, 'units_2': 6, 'units_3': 1, 'units_4': 6, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 1, 'tuner/round': 1, 'tuner/trial_id': '0017'}
{'activation': 'tanh', 'first_units': 5, 'num_layers': 4, 'units_0': 6, 'units_1': 1, 'units_2': 1, 'units_3': 6, 'units_4': 21, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}
{'activation': 'tanh', 'first_units': 9, 'num_layers': 4, 'units_0': 21, 'units_1': 26, 'units_2': 1, 'units_3': 11, 'units_4': 16, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}
```

```python
# Top 3 models
top_model = tuner.get_best_models(3)
for model in top_model:
  model_loss, model_accuracy = model.evaluate(X_test_scaled,y_test,verbose=2)
  print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')
```
```
268/268 - 1s - loss: 0.5529 - accuracy: 0.7314 - 597ms/epoch - 2ms/step
Loss: 0.5528539419174194, Accuracy: 0.7314285635948181
268/268 - 1s - loss: 0.5565 - accuracy: 0.7310 - 622ms/epoch - 2ms/step
Loss: 0.5564714074134827, Accuracy: 0.7309620976448059
268/268 - 1s - loss: 0.5583 - accuracy: 0.7304 - 639ms/epoch - 2ms/step
Loss: 0.558307409286499, Accuracy: 0.7303789854049683
```

```python
# Get best model hypreparameters.
best_hyper = tuner.get_best_hyperparameters(1)[0]
best_hyper.values
```
```
{'activation': 'tanh',
 'first_units': 9,
 'num_layers': 3,
 'units_0': 11,
 'units_1': 1,
 'units_2': 6,
 'units_3': 1,
 'units_4': 6,
 'tuner/epochs': 20,
 'tuner/initial_epoch': 7,
 'tuner/bracket': 1,
 'tuner/round': 1,
 'tuner/trial_id': '0017'}
```

```python
# Evaluate best model against full test data
best_model = tuner.get_best_models(1)[0]
model_loss, model_accuracy = best_model.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
```
268/268 - 1s - loss: 0.5529 - accuracy: 0.7314 - 901ms/epoch - 3ms/step
Loss: 0.5528539419174194, Accuracy: 0.7314285635948181
```

The best model from  kerastuner method was able to achieve a 73% accuracy rate and 55% loss which is the best we have seen yet. The top models all used the hyberbolic tangent functions and three to 4 hidden layers.

To tweak the model we would probably again use the tanh activation and potentially drop more columns to see if that could deliver higher accuracy as that achieved higher accuracy in the 2nd optimization notebook. Similarly, we could tweak the cutoff points for which we add categories to "other".