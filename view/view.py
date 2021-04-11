from pyqtgraph.Qt import QtCore
import sys
from draw_point_cloud import draw_point_cloud
from walk import draw_random_walks
from parameters import are_visible_dots
from PyQt5.QtWidgets import QApplication

# ------- plotting walks
import app
application = app.application

if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
    if are_visible_dots:
        draw_point_cloud()
    # draw_GA_walks()
    draw_random_walks()
    QApplication.instance().exec_()
