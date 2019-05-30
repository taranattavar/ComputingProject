import matplotlib.pyplot as plt

f = plt.figure()
patch = plt.Circle([-10., -10.], 1, fc='r')
ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.add_patch(patch)

for i in range(-10, 10):
    plt.Circle([-10., -10.], 1, fc='r').center = [i, i]
    plt.pause(0.01)
