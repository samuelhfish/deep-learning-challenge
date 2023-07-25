# deep-learning-challenge
Deep learning exercise using machine learning and neural networks.


All code exists in the jupyter notebooks in the "notebooks" folder. All HDF5 files are in the "h5_files" folder.

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

```
s = "Python syntax highlighting"
print s

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

```
s = "Python syntax highlighting"
print s

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

268/268 - 1s - loss: 1.2683 - accuracy: 0.5895 - 500ms/epoch - 2ms/step
Loss: 1.2683279514312744, Accuracy: 0.5895043611526489
```