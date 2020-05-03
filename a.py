import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal

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

def open_png(file_name):
    I = np.asarray(Image.open(file_name + ".png").convert('L'))
    # I = conv(I, 4)
    I = convert_image(I)
    return I

def open_phi_png():
    phi_0 = np.asarray(Image.open("c1.png").convert('L'))
    # phi_0 = conv(phi_0, 4)
    phi_0 = convert_phi(phi_0)
    return phi_0

def draw_plt(I, n = 64, m = 64):
    ## initialize of phi_0

    # phi_0 = open_phi_png()
    # phi_0 = np.eye(n)
    phi_0 = np.array([[random.random() for _ in range(n)] for _ in range(n)])


    phi_list = [phi_0.copy()]

    ## initialize of matrixes
    U_g, W_g, U_z, W_z, U_r, W_r, U_o, W_o, V = [np.eye(n) for _ in range(9)]
    b_z, b_r, b_o, b_V = [np.zeros(n) for _ in range(4)]


    ## initialize of plot shape
    rows = 3
    cols = 4
    T = rows * cols

    def add_im(I, t, title):
        ob = axes[t // cols][t % cols].imshow(I, cmap=plt.cm.gray)
        axes[t // cols][t % cols].set_title(title)
        cbar = fig.colorbar(ob, ax=axes[t // cols][t % cols], extend='both')
        cbar.minorticks_on()

    fig, axes = plt.subplots(ncols=cols, nrows=rows, sharex=True, sharey=True, figsize=(cols * 5, rows * 5))
    add_im(I, 0, "I")

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
            return -(np.abs(phi_dx_n) + np.abs(phi_dy_n))  # np.abs should be used, without it we get mars photos
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

        phi_list.append((1 - z_t) * o_t + z_t * phi_list[t - 1])  # should be trained
        phi_now = np.vectorize(lambda v: 0 if v < 0 else 1)(phi_list[t])
        add_im(phi_list[t], t, "phi_" + str(t + 1))


## reading one image
# I = open_png("nodule")
# draw_plt(I)
# plt.savefig(file_name + ".png")
# plt.show()


def read_images(file):
    I = np.asarray(Image.open(file + ".png").convert('L'))[2:-2, 2:-2]
    I_list = []
    for i in range(8):
        for j in range(8):
            I_list.append(I[i * 66: i * 66 + 64, j * 66: j * 66 + 64])
            I_list[-1] = convert_image(I_list[-1])
    return I_list

## images with many samples
file_name_1 = "individualImage"
file_name_2 = "individualImage-2"
file_name = file_name_1

list_I = read_images(file_name)

## cycle for image in list_I
for ind, I in enumerate(list_I):
    draw_plt(I)
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    plt.savefig(file_name + "/rls_test_" + str(ind) + ".png")
    plt.show()

print("done")