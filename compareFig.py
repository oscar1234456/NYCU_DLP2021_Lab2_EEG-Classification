import matplotlib.pyplot as plt
import pickle
##EEGNet Loader
with open('EEGnet_ReLU_Testing.pickle', 'rb') as f:
    EEGnetReLUTest = pickle.load(f)
with open('EEGnet_ReLU_Training.pickle', 'rb') as f:
    EEGnetReLUTrain = pickle.load(f)
with open('EEGnet_LeakyReLU_Testing.pickle', 'rb') as f:
    EEGnetLeakyReLUTest = pickle.load(f)
with open('EEGnet_LeakyReLU_Training.pickle', 'rb') as f:
    EEGnetLeakyReLUTrain = pickle.load(f)
with open('EEGnet_ELU_Testing.pickle', 'rb') as f:
    EEGnetELUTest = pickle.load(f)
with open('EEGnet_ELU_Training.pickle', 'rb') as f:
    EEGnetELUTrain = pickle.load(f)

## EEGNet Plot
epoch = 300
iter = [x+1 for x in range(epoch)]
plt.xlim(-5,305)
plt.ylim(64,102)
plt.plot(iter, EEGnetReLUTrain, 'r-', label="relu_train")
plt.plot(iter, EEGnetReLUTest, 'b-', label="relu_test")
plt.plot(iter, EEGnetLeakyReLUTrain, 'g-', label="leaky_relu_train")
plt.plot(iter, EEGnetLeakyReLUTest, 'c-', label="leaky_relu_test")
plt.plot(iter, EEGnetELUTrain, 'm-', label="elu_train")
plt.plot(iter, EEGnetELUTest, 'y-', label="elu_test")
plt.legend(loc='lower right')
plt.title("Activation function comparision(EEGNet)")
plt.show()

##DeepConvNet Loader
with open('DeepConvNet_ReLU_Testing.pickle', 'rb') as f:
    DeepConvNetReLUTest = pickle.load(f)
with open('DeepConvNet_ReLU_Training.pickle', 'rb') as f:
    DeepConvNetReLUTrain = pickle.load(f)
with open('DeepConvNet_LeakyReLU_Testing.pickle', 'rb') as f:
    DeepConvNetLeakyReLUTest = pickle.load(f)
with open('DeepConvNet_LeakyReLU_Training.pickle', 'rb') as f:
    DeepConvNetLeakyReLUTrain = pickle.load(f)
with open('DeepConvNet_ELU_Testing.pickle', 'rb') as f:
    DeepConvNetELUTest = pickle.load(f)
with open('DeepConvNet_ELU_Training.pickle', 'rb') as f:
    DeepConvNetELUTrain = pickle.load(f)

## DeepConvNet Plot
epoch = 300
iter = [x+1 for x in range(epoch)]
plt.xlim(-5,305)
plt.ylim(64,102)
plt.plot(iter, DeepConvNetReLUTrain, 'r-', label="relu_train")
plt.plot(iter, DeepConvNetReLUTest, 'b-', label="relu_test")
plt.plot(iter, DeepConvNetLeakyReLUTrain, 'g-', label="leaky_relu_train")
plt.plot(iter, DeepConvNetLeakyReLUTest, 'c-', label="leaky_relu_test")
plt.plot(iter, DeepConvNetELUTrain, 'm-', label="elu_train")
plt.plot(iter, DeepConvNetELUTest, 'y-', label="elu_test")
plt.legend(loc='lower right')
plt.title("Activation function comparision(DeepConvNet)")
plt.show()