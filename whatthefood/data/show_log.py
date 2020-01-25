import matplotlib.pyplot as plt
import numpy as np


log = 'log - Copy.txt'

tr = []
ev = []
mb = []

last_t = 0
n = 0

with open(log) as log_file:
    for line in log_file.readlines():
        tokens = line.split('\t')
        if tokens[0] == 'train':
            tr.append([last_t, *[float(t) for t in tokens[1:]]])
        elif tokens[0] == 'eval':
            ev.append([last_t, *[float(t) for t in tokens[1:]]])
        elif len(tokens) > 0:
            last_t = int(tokens[0])
            mb.append([last_t, *[float(t) for t in tokens[1:-1]]])

tr = np.array(tr).T
ev = np.array(ev).T
mb = np.array(mb).T

_, ax = plt.subplots(2, 2)

ax[0, 0].set_title('Loss')
ax[0, 0].plot(tr[0], tr[1], label='Train')
ax[0, 0].plot(ev[0], ev[1], label='Eval')

ax[0, 0].grid()
ax[0, 0].legend()

ax[0, 1].set_title('Partial losses')
ax[0, 1].plot(tr[0], tr[2], label='Train detection')
ax[0, 1].plot(tr[0], tr[3], label='Train localization')
ax[0, 1].plot(tr[0], tr[4], label='Train classification')
ax[0, 1].plot(ev[0], ev[2], label='Eval detection')
ax[0, 1].plot(ev[0], ev[3], label='Eval localization')
ax[0, 1].plot(ev[0], ev[4], label='Eval classification')

ax[0, 1].grid()
ax[0, 1].legend()

ax[1, 1].set_title('Detection')
ax[1, 1].plot(tr[0], tr[6], label='Train TP rate')
ax[1, 1].plot(ev[0], ev[6], label='Eval TP rate')
ax[1, 1].plot(tr[0], tr[7], label='Train FP rate')
ax[1, 1].plot(ev[0], ev[7], label='Eval FP rate')

ax[1, 1].grid()
ax[1, 1].legend()

ax[1, 0].set_title('Classification')
ax[1, 0].plot(tr[0], tr[9], label='Train')
ax[1, 0].plot(ev[0], ev[9], label='Eval')

ax[1, 0].grid()
ax[1, 0].legend()
plt.show()


_, ax = plt.subplots(2, 2)

ax[0, 0].plot(mb[0], mb[2])
ax[0, 0].plot(tr[0], tr[1])

ax[0, 1].plot(np.max(mb[0].reshape([-1, 2]), axis=-1), np.mean(mb[2].reshape([-1, 2]), axis=-1))
ax[0, 1].plot(tr[0], tr[1])

ax[1, 0].plot(np.max(mb[0].reshape([-1, 4]), axis=-1), np.mean(mb[2].reshape([-1, 4]), axis=-1))
ax[1, 0].plot(tr[0], tr[1])

if len(mb[2]) % 8 == 0:
    ax[1, 1].plot(np.max(mb[0].reshape([-1, 8]), axis=-1), np.mean(mb[2].reshape([-1, 8]), axis=-1))
    ax[1, 1].plot(tr[0], tr[1])
plt.show()


fig, ax = plt.subplots(1, 2)

ax[0].plot(tr[7], tr[6], zorder=1, alpha=0.5)
ax[0].scatter(tr[7], tr[6], c=tr[0], zorder=2)

ax[0].set_xlabel('Train FP rate')
ax[0].set_xlim([0, 1])
ax[0].set_ylabel('Train TP rate')
ax[0].set_ylim([0, 1])

ax[0].set_aspect('equal')

o = ax[1].plot(ev[7], ev[6], zorder=1, alpha=0.5)
ax[1].scatter(ev[7], ev[6], c=tr[0], zorder=2)

ax[1].set_xlabel('Eval FP rate')
ax[1].set_xlim([0, 1])
ax[1].set_ylabel('Eval TP rate')
ax[1].set_ylim([0, 1])

ax[1].set_aspect('equal')
plt.show()