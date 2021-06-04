import numpy as np
import cmath
from matplotlib import pyplot as plt


def f(x):
    return 10/(1+(10*x - 5)**2)


def reverse_bit(n):
    return int('{:08b}'.format(n)[::-1], 2)


def fft(f_k):
    N = len(f_k)
    if N >= 2:
        first_half = f_k[0:N//2]
        second_half = f_k[N//2:N]

        first = fft(first_half)
        second = fft(second_half)

        z = np.array([np.exp(-1j * 2 * np.pi * k / N)
                      for k in range(N//2)])

        k = z * second
        return np.append((first + k), (first - k))
    else:
        return f_k


n = 2**8

f_j = np.array([reverse_bit(x) for x in np.arange(n)])
f_j = np.array([f(x/n) for x in f_j])

f_hat = np.array(fft(f_j))/np.sqrt(n)
print(f_hat)
real = [x.real for x in f_hat]
imag = [x.imag for x in f_hat]
plt.plot(real, label='Real part')
plt.plot(imag, label='Imaginary part')
plt.legend()
plt.show()
