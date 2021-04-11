from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import sys
import pyqtgraph.opengl as gl

from parameters import view_height, elevation, azimuth, are_visible_dots
from draw_point_cloud import draw_point_cloud
from walk import draw_random_walks
app = QApplication([])
widget = gl.GLViewWidget()
widget.show()
widget.setCameraPosition(distance=view_height, elevation=elevation, azimuth=azimuth)

# ------- plotting walks

if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
    if are_visible_dots:
        draw_point_cloud(widget)
    # draw_GA_walks()
    draw_random_walks()
    QApplication.instance().exec_()
