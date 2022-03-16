from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras import regularizers

encoding_dim = 1024
hidden_dim = int(encoding_dim/2)

class Autoencoder(object):

    def __init__(self, input_dim, learning_rate = 0.005):
        self.input_dim = input_dim
        self.save_path = "autoencoder_v1.h5"
        self.learning_rate = learning_rate
        self.input = Input(shape = (self.input_dim, ))
        self.encoder = Dense(encoding_dim, activation = 'relu', activity_regularizer= regularizers.l1(learning_rate))(self.input)
        self.encoder = Dense(hidden_dim, activation='relu')(self.encoder)
        self.encoder = Dense(int(hidden_dim/2), activation = 'relu')(self.encoder)
        self.decoder = Dense(hidden_dim, activation='relu')(self.encoder)
        self.decoder = Dense(encoding_dim, activation='relu')(self.decoder)
        self.decoder = Dense(self.input_dim, activation='sigmoid')(self.decoder)

        self.autoencoder = Model(inputs = self.input, outputs = self.decoder)

        self.autoencoder.compile(metrics = ['binary_accuracy'], loss = 'binary_crossentropy',
                    optimizer = 'adam')

    def summary(self):
        #print summary of the model
        self.autoencoder.summary()

    def set_save_path(self, path):
        #change the save path
        self.save_path = path

    def fit(self, train, valid, n_epochs = 5, batch_size = 4000):
        #train the model & save the results


        cp = ModelCheckpoint(filepath=self.save_path,
                               save_best_only=True,
                               verbose=0)

        history = self.autoencoder.fit(train, train,
                            epochs=n_epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(valid, valid),
                            verbose=1,
                            callbacks=[cp]).history

        return history

    def predict(self, x):
        #predict function
        return self.autoencoder.predict(x)

    def load_model(self, path):
        #load a pre-trained model
        self.autoencoder = load_model(path)
