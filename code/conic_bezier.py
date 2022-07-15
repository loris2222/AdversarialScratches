import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])

x = np.linspace(-9, 9, 400)
y = np.linspace(-9, 9, 400)
t = np.linspace(0, 1, 100)
x, y = np.meshgrid(x, y)


def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)


def bezier(p0, p1, p2, t):
    x = (1-t)**2*p0[0] + 2*t*(1-t)*p1[0] + t**2*p2[0]
    y = (1-t)**2*p0[1] + 2*t*(1-t)*p1[1] + t**2*p2[1]
    return x, y


P0 = np.array([0, 4])
P1 = np.array([2, 2])
P2 = np.array([4, 4])

points = bezier(P0, P1, P2, t)

# plt.plot(points[0], points[1], 'go')
# plt.show()

u = np.array([P1[1]-P2[1], P2[0]-P1[0], P1[0]*P2[1]-P1[1]*P2[0]]).reshape([3, 1])
v = np.array([P2[1]-P0[1], P0[0]-P2[0], P2[0]*P0[1]-P2[1]*P0[0]]).reshape([3, 1])
w = np.array([P0[1]-P1[1], P1[0]-P0[0], P0[0]*P1[1]-P0[1]*P1[0]]).reshape([3, 1])


Q = 2*(np.matmul(u, w.transpose()) + np.matmul(w, u.transpose())) - np.matmul(v, v.transpose())
T1 = np.matmul(np.array([points[0][20], points[1][20], 1]).reshape([1, 3]), Q).reshape([3, 1])
T2 = np.matmul(np.array([points[0][80], points[1][80], 1]).reshape([1, 3]), Q).reshape([3, 1])

axes()
plt.plot(P0[0], P0[1], 'go')
plt.plot(P1[0], P1[1], 'bo')
plt.plot(P2[0], P2[1], 'ro')


def plot_line(P):
    P = P/P[2]
    a = P[0]
    b = P[1]
    c = P[2]
    plt.contour(x, y, (a*x+b*y+c), [0], colors='k')


def plot_conic(Q):
    a = Q[0, 0]
    b = 2 * Q[0, 1]
    c = Q[1, 1]
    d = 2 * Q[0, 2]
    e = 2 * Q[1, 2]
    f = Q[2, 2]

    plt.contour(x, y, (a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f), [0], colors='k')


def intersect_lines(P1, P2):
    a = P1.reshape([1, 3])
    b = P2.reshape([1, 3])
    return np.cross(a, b).reshape([3, 1])

# Plot the Bézier generated from the conic
# plot_conic(Q)
# plot_line(T1)
# plot_line(T2)
# I = intersect_lines(T1, T2)
# I = I/I[2]
# plt.plot(I[0], I[1], 'bo')
#
# plt.show()


# Split Bézier
u1 = 0.2
u2 = 0.8
PA = np.array(bezier(P0, P1, P2, u1))
P1_P = (1-u1)*P1+u1*P2
PB = np.array(bezier(P0, P1, P2, u2))
P2_P = (1-u2)*PA+u2*P1_P

points = bezier(PA, P2_P, PB, t)
plt.plot(points[0], points[1], 'go')
plt.show()

PB = bezier(P0, P1, P2, u2)
