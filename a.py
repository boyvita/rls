import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from skimage import filters
from skimage.data import camera
from skimage.util import compare_images
from scipy import misc
import glob
from PIL import Image
from skimage import filters
def normalize(A):
    return A / max(abs(min(0, A.min())), A.max())


n = 16
m = 16
k = 16
T = 150
I = np.asarray(Image.open("c.png").convert('LA'))
print(I[:, :, 0].shape)
# I = Image.fromarray()
I = I[:, :, 0]
# I.show()
a = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        a[i][j] = np.amin(I[i * 4: i * 4 + 4, j * 4: j * 4 + 4])
I = a
I = np.array([[0 if a[i][j] > 128 else 1 for j in range(n)] for i in range(n)])


phi_0 = np.asarray(Image.open("c1.png").convert('LA'))
print(phi_0[:, :, 0].shape)
phi_0 = phi_0[:, :, 0]
a = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        a[i][j] = np.amin(phi_0[i * 4: i * 4 + 4, j * 4: j * 4 + 4])
phi_0 = np.vectorize(lambda v: 1 if v > 128 else -1)(a)


# phi_0 = np.eye(n)
# phi_0 = a.transpose()
# print(I)


EPS = 1e-5



# edge_sobel = filters.sobel(phi_0)
# fig, axes = plt.subplots(ncols=5, nrows=5, sharex=True, sharey=True,
#                          figsize=(8, 8))
# axes[0].imshow(phi_0, cmap=plt.cm.gray)
# axes[0].set_title('phi_0')
# axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
# axes[1].set_title('Sobel Edge Detection')
# plt.show()
# phi_0 = np.eye(n)

U_g, W_g, U_z, W_z, U_r, W_r, U_o, W_o, V = [np.eye(n) for k in range(9)]
b_z, b_r, b_o, b_V = [np.zeros(n) for k in range(4)]


rows = 4
cols = 8
T = rows * cols

fig, axes = plt.subplots(ncols=cols, nrows=rows, sharex=True, sharey=True, figsize=(10, 10))

# phi_0 = np.array([[random.random() for j in range(m)] for i in range(n)])
phi_list = [phi_0.copy()]

axes[0][0].imshow(phi_list[0], cmap=plt.cm.gray)
axes[0][0].set_title('phi_0')

for t in range(1, T):

    def sigmoid(u):
        if u > 5:
            return 1
        if u < -5:
            return 0
        return 1 / (1 + np.exp(-u))


    def curve(phi):
        return filters.sobel(phi)


    def g(I, phi):
        c1 = 0  # average values of inside of the contour
        c2 = 0  # average values of outside of the contour
        counter = 0
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                counter += 1 if phi[i][j] < 0 else 0
                c1 += I[i][j] if phi[i][j] < 0 else 0
                c2 += I[i][j] if phi[i][j] > 0 else 0
        c1 /= counter if counter else 1
        c2 /= (n * m - counter) if (n * m - counter) else 1
        return normalize(curve(phi)) - U_g @ (I - c1) ** 2 + W_g @ (I - c2) ** 2


    x_t = g(I, phi_list[t - 1])


    z_t = np.vectorize(sigmoid)(U_z @ x_t + W_z @ phi_list[t - 1] + b_z)  # update gate zt
    r_t = np.vectorize(sigmoid)(U_r @ x_t + W_r @ phi_list[t - 1] + b_r)  # reset gate rt
    o_t = np.vectorize(np.tanh)(U_o @ x_t + W_o @ (phi_list[t - 1] * r_t) + b_o)  # intermediate hidden unit ht

    phi_list.append((1 - z_t) * o_t + z_t * phi_list[t - 1])
    # phi_list.append(o_t)


    axes[t // cols][t % cols].imshow(phi_list[t], cmap=plt.cm.gray)
    axes[t // cols][t % cols].set_title('phi_' + str(t))


    # im.save("{}.png".format(t))
plt.show()