import model

def main():
    nb_class = 2
    width, height = 224, 224


    model = SqueezeNet(
        nb_class, inputs=(3, args.height, args.width))
    dp.visualize_model(model)

    print('Build model')

    sgd = SGD(lr=0.1, decay=0.0002, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

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
