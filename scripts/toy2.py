import numpy as np
import matplotlib.pyplot as plt






from scipy import stats



def measure2(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(scale=0.5, size=n)
    m2 = np.random.normal(scale=2, size=n)
    return m1+m2, m1-m2


def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2



m1, m2 = measure(10000)
d1, d2 = measure2(10000)
xmin = m1.min()
dxmin = d1.min()
xmax = m1.max()
dxmax = d1.max()
ymin = m2.min()
dymin = d2.min()
ymax = m2.max()
dymax = d2.max()

X1, Y1 = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
X2, Y2 = np.mgrid[dxmin:dxmax:100j, dymin:dymax:100j]
positions1 = np.vstack([X1.ravel(), Y1.ravel()])
positions2 = np.vstack([X2.ravel(), Y2.ravel()])
values1 = np.vstack([m1, m2])
values2 = np.vstack([d1, d2])
kernel1 = stats.gaussian_kde(values1)
kernel2 = stats.gaussian_kde(values2)
# print(kernel(positions))
Z1 = np.reshape(kernel1(positions1).T, X1.shape)
Z2 = np.reshape(kernel2(positions2).T, X2.shape)
# print(Z)

Z = np.multiply(Z1, Z2)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.rot90(Z2), cmap=plt.cm.Reds,
            extent=[dxmin, dxmax, dymin, dymax], alpha = 0.5)
ax.imshow(np.rot90(Z1), cmap=plt.cm.Blues,
          extent=[xmin, xmax, ymin, ymax], alpha = 0.5)


ax.imshow(np.rot90(Z), cmap=plt.cm.Greens,
          extent=[xmin, xmax, ymin, ymax], alpha = 0.5)

# ax.plot(m1, m2, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()
