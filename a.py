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
from scipy import signal
from scipy import misc


EPS = 1e-5



def normalize(A):
    return A / max(abs(min(0, A.min())), A.max())



def conv(input, step):
    n = input.shape[0] // step
    m = input.shape[1] // step
    ans = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            ans[i][j] = np.amin(input[i * step: i * step + step, j * step: j * step + step])
    return ans

def convert_image(input):
    return np.vectorize(lambda v: v / 255)(input)


def convert_phi(input):
    return np.vectorize(lambda v: 1 - v / 127.5)(input)

def open_c_png():
    I = np.asarray(Image.open("c.png").convert('L'))
    # I = conv(I, 4)
    I = convert_image(I)
    return I

def open_nodule_png():
    I = np.asarray(Image.open("nodule.png").convert('L'))
    # I = conv(I, 4)
    I = convert_image(I)
    return I

def open_phi_png():
    phi_0 = np.asarray(Image.open("c1.png").convert('L'))
    # phi_0 = conv(phi_0, 4)
    phi_0 = convert_phi(phi_0)
    return phi_0

I = open_c_png()
# I = open_nodule_png()
phi_0 = open_phi_png()
n = I.shape[0]
m = I.shape[1]
T = 150

# phi_0 = np.eye(n)
# phi_0 = np.array([[random.random() for _ in range(n)] for _ in range(n)])


def read_images():
    I = np.asarray(Image.open("individualImage.png").convert('L'))[2:-2, 2:-2]
    I_list = []
    for i in range(8):
        for j in range(8):
            I_list.append(I[i * 66: i * 66 + 64, j * 66: j * 66 + 64])
            I_list[-1] = convert_image(I_list[-1])
    return I_list

list_I = read_images()

for ind, I in enumerate(list_I):
    if ind > 10:
        break
    phi_list = [phi_0.copy()]

    U_g, W_g, U_z, W_z, U_r, W_r, U_o, W_o, V = [np.eye(n) for _ in range(9)]
    # U_z /= 8
    # U_g /= 10
    # U_r /= 8
    # U_o /= 10
    b_z, b_r, b_o, b_V = [np.zeros(n) for _ in range(4)]

    rows = 3
    cols = 4
    T = rows * cols

    fig, axes = plt.subplots(ncols=cols, nrows=rows, sharex=True, sharey=True, figsize=(cols * 5, rows * 5))
    axes[0][0].imshow(I, cmap=plt.cm.gray)
    axes[0][0].set_title('I')


    for t in range(1, T):
        def sigmoid(u):
            if u > 5:
                return 1
            if u < -5:
                return 0
            return 1 / (1 + np.exp(-u))

        SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        def img_derivative(input, sobel_kernel):
            return signal.convolve2d(input, sobel_kernel, boundary='symm', mode="same")

        def curve(phi):
            phi_dx = img_derivative(phi, SOBEL_X)
            phi_dy = img_derivative(phi, SOBEL_Y)

            phi_dx_n = phi_dx / (phi_dx ** 2 + phi_dy ** 2 + EPS) ** 0.5
            phi_dy_n = phi_dy / (phi_dx ** 2 + phi_dy ** 2 + EPS) ** 0.5

            # return -(phi_dx_n + phi_dy_n) # plain formula

            # return filters.sobel(phi) # it works odd, but fairly good

            # return -(phi_dx_n + phi_dy_n) # np.abs should be used, without it we get mars photos
            return -(np.abs(phi_dx_n) + np.abs(phi_dy_n)) # np.abs should be used, without it we get mars photos
            #
            # return normalize(-(np.abs(phi_dx_n) + np.abs(phi_dy_n))) # it should be normalized, it is visible with I=c_png


        def g(I, phi):
            c1 = 0  # average values of inside of the contour
            c2 = 0  # average values of outside of the contour
            counter = 0
            for i in range(phi.shape[0]):
                for j in range(phi.shape[1]):
                    counter += 1 if phi[i][j] > 0 else 0
                    c1 += I[i][j] if phi[i][j] > 0 else 0
                    c2 += I[i][j] if phi[i][j] < 0 else 0
            c1 /= counter if counter else 1
            c2 /= (n * m - counter) if (n * m - counter) else 1
            return curve(phi) - U_g @ (10 * I - c1) ** 2 + W_g @ (10 * I - c2) ** 2


        x_t = g(I, phi_list[t - 1])


        z_t = np.vectorize(sigmoid)(U_z @ x_t + W_z @ phi_list[t - 1] + b_z)  # update gate zt
        r_t = np.vectorize(sigmoid)(U_r @ x_t + W_r @ phi_list[t - 1] + b_r)  # reset gate rt
        o_t = np.vectorize(np.tanh)(U_o @ x_t + W_o @ (phi_list[t - 1] * r_t) + b_o)  # intermediate hidden unit ht

        phi_list.append((1 - z_t) * o_t + z_t * phi_list[t - 1]) # should be trained
        # phi_list.append(z_t * phi_list[t - 1]) # temporary, focus only what inside phi
        phi_now = np.vectorize(lambda v: 0 if v < 0 else 1)(phi_list[t])
        axes[t // cols][t % cols].imshow(phi_now, cmap=plt.cm.gray)
        # axes[t // cols][t % cols].imshow(phi_list[t], cmap=plt.cm.gray)
        axes[t // cols][t % cols].set_title('phi_' + str(t))

    plt.savefig("rls_test_" + str(ind + 1) + ".png")
    plt.show()

print("done")