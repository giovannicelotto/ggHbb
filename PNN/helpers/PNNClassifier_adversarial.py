import numpy as np
import tensorflow as tf
from getFeatures import getFeatures
from getModel import getModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.layers import BatchNormalization
from doPlots import doPlotLoss

def PNNClassifier_adversarial(Xtrain, Xtest, Ytrain, Ytest, advFeature_train, advFeature_test, Wtrain, Wtest, rWtrain, rWtest, featuresForTraining, hp, inFolder, outFolder):
    
    class GradientReversalLayer(Layer):
        def __init__(self):
            super(GradientReversalLayer, self).__init__()

        def call(self, inputs):
            @tf.custom_gradient
            def reverse_gradient(x):
                def grad(dy):
                    return -dy  # Reverse the gradient sign
                return x, grad
            return reverse_gradient(inputs)

    # Step 3: Define the models
    # Classifier model
    def build_classifier(input_shape):
        inputs = Input(shape=input_shape)
        x = Dense(16, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(4, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(2, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs, name="Classifier")

    # Adversary model
    def build_adversary():
        inputs = Input(shape=(1,))  # Takes classifier scores as input
        x = GradientReversalLayer()(inputs)  # Reverse the gradients
        x = Dense(8, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(4, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(2, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)  # Predict `advFeature`
        return Model(inputs, outputs, name="Adversary")
    modelName = "myModel.h5"
    input_shape = (Xtrain.shape[1],)
    classifier = build_classifier(input_shape)
    adversary = build_adversary()

    optimizer = Adam(learning_rate = 5e-4, name="Adam") 
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy')
    classifier.fit(Xtrain, Ytrain, batch_size=32, epochs=5, validation_data=(Xtest, Ytest))

    inputs = Input(shape=input_shape)
    classifier_scores = classifier(inputs)
    adversary_output = adversary(classifier_scores)

    combined_model = Model(inputs=inputs, outputs=[classifier_scores, adversary_output])
    print(combined_model.summary())

    # Step 5: Compile the combined model
    classification_loss = tf.keras.losses.BinaryCrossentropy()
    adversarial_loss = tf.keras.losses.MeanAbsoluteError()
    lambda_penalty = .25  # Adjust this hyperparameter
    #optimizer = Adam(learning_rate = 5e-5, name="Adam") 
    combined_model.compile(
        optimizer='adam',
        loss=[classification_loss, adversarial_loss],
        loss_weights=[1.0, lambda_penalty]
    )
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the total validation loss
        patience=50,         # Stop after 10 epochs with no improvement
        restore_best_weights=True  # Restore the weights from the best epoch
    )

    # Step 6: Train the model
    fit = combined_model.fit(
        Xtrain, [Ytrain, advFeature_train],
        validation_data=(Xtest, [Ytest, advFeature_test]),
        batch_size=32,
        epochs=2000,
        verbose=1,
        callbacks=[early_stopping] 

    )
    model.save(outFolder +"/model/"+modelName)
    model = load_model(outFolder +"/model/"+modelName)
    doPlotLoss(fit=fit, outName=outFolder +"/performance/loss.png", earlyStop=earlyStop, patience=hp['patienceES'])

    YPredTest = model.predict(Xtest[featuresForTraining])
    YPredTrain = model.predict(Xtrain[featuresForTraining])


    return YPredTrain, YPredTest, model, featuresForTraining