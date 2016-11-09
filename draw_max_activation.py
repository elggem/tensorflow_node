"""
http://deeplearning.stanford.edu/wiki/index.php/Visualizing_a_Trained_Autoencoder
"""

import matplotlib.pyplot as plt

W = model.weights[0]

outputs = []

#calculate for each hl
for i in xrange(W.shape[1]):
    output = np.array(np.zeros(W.shape[0]),dtype='float32')

    W_ij_sum = 0
    for j in xrange(output.size):
        W_ij_sum += np.power(W[j][i],2)

    for j in xrange(output.size): 
        W_ij = W[j][i]
        output[j] = (W_ij)/(np.sqrt(W_ij_sum))

    outputs.append(output)

#plot the results
f, a = plt.subplots(10, 10, figsize=(10, 10))

for i in range(10):
    for j in range(10):
        a[i][j].imshow(outputs[((i+1)*(j+1))-1].reshape([28,28]), cmap='Greys', interpolation="nearest")

f.show()
plt.draw()
plt.waitforbuttonpress()


