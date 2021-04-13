import sys

from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore

# ------- plotting walks
# import app
from view.draw_point_cloud import draw_point_cloud
from view.parameters import are_visible_dots
from view.walk import draw_genetic_algorithm_walks

def run_view():
    # application = app.application

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        if are_visible_dots:
            draw_point_cloud()
        draw_genetic_algorithm_walks()
        # draw_random_walks()
        QApplication.instance().exec_()

run_view()
