import model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def main():
    np.random.seed(44)
    nb_class = 2
    width, height = 224, 224


    sn = model.SqueezeNet(nb_classes=nb_class, inputs=(3, height, width))

    print('Build model')

    sgd = SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True)
    sn.compile(
        optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 800
    nb_epoch = 1000

    #   Generator
    #train_datagen = ImageDataGenerator(
    #        rescale=1./255,
    #        shear_range=0.2,
    #        zoom_range=0.2,
    #        horizontal_flip=True)
    train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(width, height),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(width, height),
            batch_size=32,
            class_mode='categorical')

    sn.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)

    sn.save_weights('weights.h5')



    # Evaluate


    # Predict


    # Save Weights

#    print("Training...")
#    model.fit_generator(
#        train_generator,
#        samples_per_epoch=nb_train_samples,
#        nb_epoch=args.epochs,
#        validation_data=validation_generator,
#        nb_val_samples=nb_val_samples)

#    print "[squeezenet] Model trained."

#    t0 = tl.print_time(t0, 'score squeezenet')
#    model.save_weights('squeeze_net.h5', overwrite=True)

if __name__ == '__main__':
    main()
