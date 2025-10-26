import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle

data = pd.read_csv('churn_data.csv')
print(data.head())

##prepreocess the data
#drop irrelevant columns/features
data =  data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) #axis=1 means drop columns
print(data.head())  

##geography and gender are categorical features, convert them to numerical 
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
print(data.head())
    
#one hot encoding for geography as it has more than 2 categories
one_hot_encoder = OneHotEncoder()
geography_encoded = one_hot_encoder.fit_transform(data[['Geography']])
print(geography_encoded.toarray())

#print(one_hot_encoder.get_feature_names_out(['Geography']))

geo_encoded_df = pd.DataFrame(geography_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(['Geography'])) 
print(geo_encoded_df.head())

#combine the encoded geography columns with the original data
data = pd.concat([data, geo_encoded_df], axis=1)   
data = data.drop(['Geography'], axis=1)  #drop the original geography column
print(data.head())  

#save the encoders for future use
with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)

with open('one_hot_encoder_geography.pkl', 'wb') as f:
    pickle.dump(one_hot_encoder, f) 

#divide the dataset into dependent and independent features
#exited variable is 'Exited'(o/p)column is dependent variable while rest are independent features
x = data.drop(['Exited'], axis=1) #independent features
y = data['Exited']  #dependent feature

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#save the scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


########creating ANN model using keras##########
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

#build the model
#number of i.p features = number of columns in x_train which is x_train.shape[1]
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),  #Hidden layer 1, input layer is same as number of features we have and i/p layer is not explicitly defined and its connected to HL1
    Dense(32, activation='relu'),  #Hidden layer 2
    Dense(1, activation='sigmoid')  #output layer

])


#print(model.summary()) , 2945 in total params is nothing but number of weights and biases in the model

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#here, adam is an optimizer which is used to update the weights during backpropagation and has fixed learning rate
#to give your own learning rate, use tf.keras.optimizers.Adam(learning_rate=0.01) instead of 'adam'
#similary, loss function can also be initialised using tf.keras.losses.BinaryCrossentropy() instead of 'binary_crossentropy'
#binary_crossentropy is used for binary classification problems
#for multi class classification, use categorical_crossentropy

##setup the tensorboard , for storing the training logs
#tensorboard is for visualizing the training process , can also use matplotlib for plotting graphs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

##setup early stopping , to stop training when the model stops improving on validation loss
#let's say we are running for 100 epochs but after 20 epochs the model stops improving on validation loss
#so we can stop the training at that point to save time and resources   
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#patience is number of epochs to wait before stopping the training after no improvement

#train the model
#history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
 #                   callbacks=[tensorboard_callback, early_stopping_callback])

history=model.fit(
        x_train,y_train,validation_data=(x_test, y_test),
        epochs=100,callbacks=[early_stopping_callback, tensorboard_callback]
)
#here, validation_split=0.2 means 20% of training data will be used for validation
#batch_size is number of samples per gradient update, default is 32
#validation_data is used to evaluate the model on test data after each epoch, y_test is true labels for x_test
#callbacks are used to monitor the training process and perform actions like early stopping, tensorboard logging etc.

#save the model
model.save('churn_model.h5')
#h5 is a file format for storing large amounts of numerical data
#use of .save() method saves the architecture, weights and training configuration of the model in a single file
#to load the model later, use tf.keras.models.load_model('churn_model.h5')


#load tensorboard to visualize the training process
#run this command in terminal/command prompt
#tensorboard --logdir=logs/fit
#then open the url http://localhost:6006/ in your browser to see the tensor