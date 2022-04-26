import os
import sys
import time

import numpy as np
from collections import deque

from numpy.lib.function_base import average
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide2'
from PySide2 import QtWidgets
import pyqtgraph as pg

max_queue_len = 50

legend_a = f"mean reward (last {max_queue_len})"
legend_b = "epsilon (x10)"
legend_c = "episode rewards"

app = QtWidgets.QApplication(sys.argv)


plt = pg.plot()
pg.setConfigOption('foreground', (0, 0, 0))
my_brush = pg.mkBrush('k', width=3)
default_brush = plt.foregroundBrush()


plt.setWindowTitle('RL reward')
plt.setForegroundBrush(my_brush)
plt.addLegend()
plt.setForegroundBrush(default_brush)

plt.show()
plt.setBackground((200, 200, 200))

plt.setLabel('bottom', 'episodes')


while True:

    queue = deque(maxlen=max_queue_len)
    x = []
    a = []
    b = []
    c = []
    plt.clear()

    zero_marker = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((130, 130, 130), width=2), movable=False)
    zero_marker.setValue(0.)
    zero_marker.setZValue(0)
    plt.addItem(zero_marker)


    with open('output.csv', 'r') as f:
        episode = 0
        for l in f.readlines():
            episode += 1
            v = l.strip().split(',')

            queue.append(float(v[0]))
            npa = np.average(np.array(queue, dtype=np.float64))
            x.append(episode)
            a.append(npa)
            b.append(10.*float(v[1]))
            c.append(float(v[0]))

    plt.plot(x, b, pen=pg.mkPen('r', width=2), name=legend_b)

    my_brush = pg.mkBrush('k', width=3)
    plt.plot(x, c, pen=None, name=legend_c, symbol='+', symbolSize=10, symbolWidth=30, brush=my_brush)
    plt.plot(x, a, pen=pg.mkPen('b', width=2), name=legend_a)

    v = np.array(a).argmax()
    marker = pg.InfiniteLine(pos=v, angle=90, pen=pg.mkPen((220, 140, 100), width=2), movable=False)
    marker.setValue(v)
    marker.setZValue(-10)
    plt.addItem(marker)
    marker = pg.InfiniteLine(pos=max(a), angle=0, pen=pg.mkPen((220, 140, 100), width=2), movable=False)
    marker.setValue(max(a))
    marker.setZValue(-10)
    plt.addItem(marker)

    for _ in range(30):
        app.processEvents()
        time.sleep(0.1)

