import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def Hair_g(beta, theta):
    return np.exp(-0.5 * theta * theta / (beta * beta)) / (np.sqrt(2 * np.pi) * beta)

def GaussianDetector(beta, phi):
    Dp = 0.0
    for k in range(-4, 4):
        Dp += Hair_g(beta, phi - (2 * np.pi) * float(k))
    return Dp

def Omega(h, eta_prime):
    gamma_i = np.arcsin(h)
    gamma_t = np.arcsin(h / eta_prime)
    return 2 * gamma_t - 2 * gamma_i + np.pi

def AzimuthalScattering(beta, phi, thetaD):
    cosThetaD = np.cos(thetaD)

    eta = 1.55
    eta_prime = np.sqrt(eta * eta - 1 + cosThetaD * cosThetaD) / cosThetaD

    SampleCnt = 16
    Dtt = 0.0
    for i in range(SampleCnt):
        h = float(i) / float(SampleCnt - 1) * 2.0 - 1.0
        h = max(min(h, 1.0), -1.0)
        Dtt += GaussianDetector(beta, phi - Omega(h, eta_prime))
    Dtt *= 2.0 / float(SampleCnt)
    return 0.5 * Dtt

def GaussianFitting(Xs, Ys):
    lnY = np.log(Ys)

    mY = np.matrix(lnY).getT()
    mX = np.matrix([[1.0 for i in range(len(Xs))], Xs, np.multiply(Xs, Xs).tolist()]).getT()

    mB = (mX.getT() * mX).getI() * mX.getT() * mY

    b0 = mB.tolist()[0][0]
    b1 = mB.tolist()[1][0]
    b2 = mB.tolist()[2][0]

    a = np.exp(b0 - b1 * b1 / (4 * b2))
    b = -1 / b2
    c = -b1 / (2 * b2)
    return a, b, c

# def GaussianFitting2(Xs, Ys):
#     lnY = np.log(Ys)
#
#     mY = np.matrix(lnY).getT()
#     mX = np.matrix([[1.0 for i in range(len(Xs))], Xs]).getT()
#
#     mB = (mX.getT() * mX).getI() * mX.getT() * mY
#
#     b0 = mB.tolist()[0][0]
#     b1 = mB.tolist()[1][0]
#     b2 = mB.tolist()[2][0]
#
#     a = np.exp(b0 - b1 * b1 / (4 * b2))
#     b = -1 / b2
#     c = -b1 / (2 * b2)
#     return a, b, c


# output image
w, h = 4, 4
phi_sample_count = 64

fig, axs = plt.subplots(w, h, figsize=(w * 3, h * 2), sharex='all', sharey='all')

for v in range(h):
    for u in range(w):
        thetaD = float(u) * np.pi / 2.0 / float(w - 1)
        beta = 0.3 + float(v) / float(h - 1) * 0.6     # 0.1 - 0.9

        Xs = []
        Ys = []
        for p in range(phi_sample_count):
            phi = float(p) / float(phi_sample_count - 1) * 2.0 * np.pi
            Dtt = AzimuthalScattering(beta, phi, thetaD)
            Xs.append(phi)
            Ys.append(Dtt)

        a, b, c = GaussianFitting(Xs, Ys)
        gx = np.arange(0,2.0 * np.pi,0.2)
        gy = a * np.exp(-np.square(gx-c)/b)

        axs[u, v].plot(Xs, Ys, 'ro', markersize=2)
        axs[u, v].plot(gx, gy)
        title = 'theta:' + '{:.1f}'.format(thetaD) + ' beta:' + '{:.1f}'.format(beta)
        title += '\namp:' + '{:.2f}'.format(a) + ' std:' + '{:.2f}'.format(b) + ' mea:' + '{:.2f}'.format(c)
        axs[u, v].set_title(title, horizontalalignment='left', x=0.0, y=1.0)

for ax in axs.flat:
    ax.set(xlabel='phi', ylabel='D_TT')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# plt.figure().set_figheight(2)
fig.tight_layout(pad=1.5)
plt.show()

# test phi graph
# u = 0.0
# v = 0.0
# phi_sample_count = 64
# theta = u * np.pi
# beta = 0.4 + v * 0.5  # 0.4 - 0.9
#
# Xs = []
# Ys = []
# for p in range(phi_sample_count):
#     phi = float(p) * 2 * np.pi / (phi_sample_count - 1)
#     Dtt = AzimuthalScattering(beta, phi, theta)
#     Xs.append(phi)
#     Ys.append(Dtt)
#
# plt.plot(Xs, Ys, 'ro')
# # plt.axis([0, 650, 0, 350])
# plt.xlabel('phi')
# plt.ylabel('Dtt')
# plt.show()
######################################

# output image
# w, h = 32, 32
# phi_sample_count = 16
#
# newimdata = []
# for v in range(h):
#     for u in range(w):
#         theta = float(u) * np.pi / (w - 1)
#         beta = 0.1 + float(v) / (h - 1) * 0.8     # 0.1 - 0.9
#
#         Xs = []
#         Ys = []
#         for p in range(phi_sample_count):
#             phi = float(p) * 2 * np.pi / (phi_sample_count - 1)
#             Dtt = AzimuthalScattering(beta, phi, theta)
#             Xs.append(phi)
#             Ys.append(Dtt)
#
#         a, b, c = GaussianFitting(Xs, Ys)
#         newimdata.append((a, b, c))
#
# img = Image.new("RGB", (w, h))
# img.putdata(newimdata)
# img.show()
#######################################################



# graph example
# pos = [[200, 300, 400, 440, 500, 600], [200, 280, 320, 310, 280, 220]]
#
# lnY = np.log(pos[1])
# print('lnY ', lnY)
#
# mY = np.matrix(lnY).getT()
# mX = np.matrix([[1.0 for i in range(len(pos[1]))], pos[0], np.multiply(pos[0], pos[0]).tolist()]).getT()
# print('mX ', mX)
# print('mY ', mY)
#
# mB = (mX.getT() * mX).getI() * mX.getT() * mY
# print('mB ', mB)
#
# b0 = mB.tolist()[0][0]
# b1 = mB.tolist()[1][0]
# b2 = mB.tolist()[2][0]
#
# print('b0 ', b0)
# print('b1 ', b1)
# print('b2 ', b2)
#
# print('b0 - b1 * b1 / (4 * b2) ', b0 - b1 * b1 / (4 * b2))
# a = np.exp(b0 - b1 * b1 / (4 * b2))
# b = -1 / b2
# c = -b1 / (2 * b2)
# print('a ', a)
# print('b ', b)
# print('c ', c)
#
# x = np.arange(0,650,1)
# y = a * np.exp(-np.square(x-c)/b)
# # print('x ', x)
# # print('y ', y)
#
# plt.plot(pos[0], pos[1], 'ro')
# plt.plot(x,y)
# plt.axis([0, 650, 0, 350])
# # plt.ylabel('gaussian distribution')
# plt.show()
###########################################################