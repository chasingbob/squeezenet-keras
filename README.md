# squeezenet-keras
SqueezeNet Keras Dogs vs. Cats demo

I'm supprised SqueezeNet has generated so little hype in the media - so I have decided to try it out for myself.

SqueezeNet's claim to fame is getting AlexNet level accuracy using far less parameters. In this demo I put it to the test. I used the Keras implementation of the Dogs vs. Cats 
[demo](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-editio) by Francois Chollet as base to see for myself.

I have spent almost no time in trying to find the optimal hyper parameters and have intiuitively gone for SGD for a small decay factor. The graph below show steady training before 
the training accuracy seperates from the validation accuracy peaking at around +/-0.8 accuracy. Pretty impressive on a small data set (2000 training images). 

![SqueezeNet Training](training_acc_loss.png)

What I am even more interested in is in the size and complexity of the model. 
The Keras implemented model has ~736000 parameters and takes up 3MB in diskspace when saving weights only. To put that in comparison, a Keras VGG16 implementation I use 
often has 138000000 parameters and weights takes up over 500MB in space. 




References.

[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and ,0.5MB model size](https://arxiv.org/abs/1602.07360)

[Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

[Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
