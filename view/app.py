from PyQt5.QtWidgets import QApplication
import pyqtgraph.opengl as gl

from parameters import view_height, elevation, azimuth


application = QApplication([])
widget = gl.GLViewWidget()
widget.show()
widget.setCameraPosition(distance=view_height, elevation=elevation, azimuth=azimuth)
