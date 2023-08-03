import numpy as np
import matplotlib.pyplot as plt

#ACC_TRAIN
Accuracy_train1 = np.load("/home/nick/CL05/ImportanceSampling-ex05/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_train.npy")
Accuracy_train2 = np.load("/home/nick/CL10/ImportanceSampling-ex10/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_train.npy")
Accuracy_train3 = np.load("/home/nick/CL11/ImportanceSampling-ex11/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_train.npy")
Accuracy_train4 = np.load("/home/nick/CL12/ImportanceSampling-ex12/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_train.npy")
# Plot each line with a different color
plt.plot(Accuracy_train1, color='blue', label='Case 1')
plt.plot(Accuracy_train2, color='red', label='Case 2')
plt.plot(Accuracy_train3, color='green', label='Case 3')
plt.plot(Accuracy_train4, color='orange', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.title('Train accuracy per Epoch for 4 cases of changing M under CIFAR-10 and budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/plotc10b2/resultsforacctrain.png")

#ACC_TEST
Accuracy_test1 = np.load("/home/nick/CL05/ImportanceSampling-ex05/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_val.npy")
Accuracy_test2 = np.load("/home/nick/CL10/ImportanceSampling-ex10/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_val.npy")
Accuracy_test3 = np.load("/home/nick/CL11/ImportanceSampling-ex11/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_val.npy")
Accuracy_test4 = np.load("/home/nick/CL12/ImportanceSampling-ex12/metrics_unif-SGD_cifar10_randaugment_b02/accuracy_per_epoch_val.npy")
# Plot each line with a different color
plt.plot(Accuracy_test1, color='blue', label='Case 1')
plt.plot(Accuracy_test2, color='red', label='Case 2')
plt.plot(Accuracy_test3, color='green', label='Case 3')
plt.plot(Accuracy_test4, color='orange', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.title('Test accuracy per Epoch for 4 cases of changing M under CIFAR-10 and budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/plotc10b2/resultsforacctest.png")

#LOSS_TRAIN
Loss_train1 = np.load("/home/nick/CL05/ImportanceSampling-ex05/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_train.npy")
Loss_train2 = np.load("/home/nick/CL10/ImportanceSampling-ex10/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_train.npy")
Loss_train3 = np.load("/home/nick/CL11/ImportanceSampling-ex11/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_train.npy")
Loss_train4 = np.load("/home/nick/CL12/ImportanceSampling-ex12/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_train.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='orange', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss per Epoch for 4 cases of changing M under CIFAR-10 and budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/plotc10b2/resultsforlosstrain.png")

#LOSS_TEST
Loss_test1 = np.load("/home/nick/CL05/ImportanceSampling-ex05/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_val.npy")
Loss_test2 = np.load("/home/nick/CL10/ImportanceSampling-ex10/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_val.npy")
Loss_test3 = np.load("/home/nick/CL11/ImportanceSampling-ex11/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_val.npy")
Loss_test4 = np.load("/home/nick/CL12/ImportanceSampling-ex12/metrics_unif-SGD_cifar10_randaugment_b02/LOSS_epoch_val.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='orange', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Test loss per Epoch for 4 cases of changing M under CIFAR-10 and budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/plotc10b2/resultsforlosstest.png")



