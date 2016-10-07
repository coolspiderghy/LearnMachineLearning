# Create first network with Keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(8,12,init='uniform'))
model.add(Activation('relu'))
model.add(Dense(12,8, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(8,1, init='uniform'))
model.add(Activation('sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')
# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print(scores*100)
print X.shape,Y.shape,X,Y,type(X),type(Y)