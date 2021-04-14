from PyQt5.QtWidgets import QApplication
import pyqtgraph.opengl as gl

from view.parameters import view_height, elevation, azimuth, background_color

application = QApplication([])
widget = gl.GLViewWidget()
widget.show()
widget.setCameraPosition(distance=view_height, elevation=elevation, azimuth=azimuth)
widget.setBackgroundColor(background_color)
widget.setFixedWidth(550)
widget.setFixedHeight(550)