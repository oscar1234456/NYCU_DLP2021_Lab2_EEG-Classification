import numpy as np
import matplotlib.pyplot as plt
# f = lambda x: np.maximum(0,x)

# x = np.arange(-5.0, 6.0, 1)
# y = f(x)
#
# plt.plot(x, y, label="ReLu")
# plt.ylim(-0.5, 6)
# plt.xlim(-6, 6)
# plt.title("ReLu")
# plt.plot([0,0],[6,-0.5], 'k--')
# plt.legend(loc='lower right')
# plt.show()

def elu(z,alpha):
	return z if z >= 0 else alpha*(np.exp(z) -1)


def plot(px, py):
    plt.plot(px, py, label="ELU")
    plt.title("ELU (alpha=0.3)")
    plt.plot([0,0],[6,-0.5], 'k--')
    plt.plot([-6, 6], [0, 0], 'k--')
    plt.ylim(-0.5, 6)
    plt.xlim(-6, 6)
    plt.legend(loc='upper left')
    plt.show()


def main():
    # Init
    a = 0.3
    x = []
    dx = -20
    while dx <= 20:
        x.append(dx)
        dx += 0.1

    # px and py
    px = [xv for xv in x]
    py = [elu(xv, a) for xv in x]

    # Plot
    plot(px, py)


if __name__ == "__main__":
    main()