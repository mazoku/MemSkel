import os.path
import numpy as np
import cv2

try:
    from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QT_VERSION_STR
    from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QBrush, QColor
    from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog
except ImportError:
    try:
        from PyQt4.QtCore import Qt, QRectF, pyqtSignal, QT_VERSION_STR
        from PyQt4.QtGui import QGraphicsView, QGraphicsScene, QImage, QPixmap, QPainterPath, QFileDialog, QBrush, QColor
    except ImportError:
        raise ImportError("ImageViewerQt: Requires PyQt5 or PyQt4.")


__author__ = "Marcel Goldschen-Ohm <marcel.goldschen@gmail.com>"
__version__ = '0.9.0'


class ImageViewerQt(QGraphicsView):
    """ PyQt image viewer widget for a QPixmap in a QGraphicsView scene with mouse zooming and panning.
    Displays a QImage or QPixmap (QImage is internally converted to a QPixmap).
    To display any other image format, you must first convert it to a QImage or QPixmap.
    Some useful image format conversion utilities:
        qimage2ndarray: NumPy ndarray <==> QImage    (https://github.com/hmeine/qimage2ndarray)
        ImageQt: PIL Image <==> QImage  (https://github.com/python-pillow/Pillow/blob/master/PIL/ImageQt.py)
    Mouse interaction:
        Left mouse button drag: Pan image.
        Right mouse button drag: Zoom box.
        Right mouse button doubleclick: Zoom to show entire image.
    """

    # Mouse button signals emit image scene (x, y) coordinates.
    # !!! For image (row, column) matrix indexing, row = y and column = x.
    leftMouseButtonPressed = pyqtSignal(float, float)
    mouseMoved = pyqtSignal(float, float, Qt.MouseButton)
    # rightMouseButtonPressed = pyqtSignal(float, float)
    # leftMouseButtonReleased = pyqtSignal(float, float)
    # rightMouseButtonReleased = pyqtSignal(float, float)
    # leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    # rightMouseButtonDoubleClicked = pyqtSignal(float, float)

    def __init__(self):
        QGraphicsView.__init__(self)

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        self.scene = QGraphicsScene()
        # self.scene.setBackgroundBrush(QBrush(Qt.black))
        self.setScene(self.scene)

        # Store a local handle to the scene's current image pixmap.
        self._pixmapHandle = None

        # Image aspect ratio mode.
        # !!! ONLY applies to full image. Aspect ratio is always ignored when zooming.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio

        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.marking = False
        self.lastPoint = None


    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def clearImage(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None

    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None

    def image(self):
        """ Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None

    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()

    def loadImageFromFile(self, fileName=""):
        """ Load an image from file.
        Without any arguments, loadImageFromFile() will popup a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """
        if len(fileName) == 0:
            if QT_VERSION_STR[0] == '4':
                fileName = QFileDialog.getOpenFileName(self, "Open image file.")
            elif QT_VERSION_STR[0] == '5':
                fileName, dummy = QFileDialog.getOpenFileName(self, "Open image file.")
        if len(fileName) and os.path.isfile(fileName):
            image = QImage(fileName)
            self.setImage(image)

    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        # if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
        #     self.fitInView(self.zoomStack[-1], Qt.IgnoreAspectRatio)  # Show zoomed rect (ignore aspect ratio).
        # else:
        #     self.zoomStack = []  # Clear the zoom stack (in case we got here because of an invalid zoom).
        self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).

    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.updateViewer()

    def mousePressEvent(self, event):
        """ Start mouse pan or zoom mode.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        # elif event.button() == Qt.RightButton:
        #     self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())
        # QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        # if event.buttons() & Qt.LeftButton:# and self.marking:
        scenePos = self.mapToScene(event.pos())
        self.mouseMoved.emit(scenePos.x(), scenePos.y(), event.buttons())

    def mouseReleaseEvent(self, event):
        """ Stop mouse pan or zoom mode (apply zoom if valid).
        """
        pass
        # QGraphicsView.mouseReleaseEvent(self, event)
        # scenePos = self.mapToScene(event.pos())
        # if event.button() == Qt.LeftButton:
        #     self.setDragMode(QGraphicsView.NoDrag)
        #     self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
        # elif event.button() == Qt.RightButton:
            # self.setDragMode(QGraphicsView.NoDrag)
            # self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        pass
        # scenePos = self.mapToScene(event.pos())
        # if event.button() == Qt.LeftButton:
        #     self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        # elif event.button() == Qt.RightButton:
        #     if self.canZoom:
        #         self.zoomStack = []  # Clear zoom stack.
        #         self.updateViewer()
        #     self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        # QGraphicsView.mouseDoubleClickEvent(self, event)


if __name__ == '__main__':
    import sys
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt4.QtGui import QApplication
        except ImportError:
            raise ImportError("ImageViewerQt: Requires PyQt5 or PyQt4.")
    print('ImageViewerQt: Using Qt ' + QT_VERSION_STR)


    class Drawer:

        def __init__(self):
            self.last_pt = None

            # viewer.loadImageFromFile()  # Pops up file dialog.
            fname = 'data/icons/kky.png'
            self.image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            self.qimage = QImage(fname)

            self.app = QApplication(sys.argv)

            # Create image viewer and load an image file to display.
            self.viewer = ImageViewerQt()

            self.viewer.setImage(self.qimage)

            self.seeds = np.zeros(self.image.shape[:2], dtype=np.uint8)
            # self.red = np.zeros(self.image.shape, dtype=np.uint8)
            # self.red[:, :, 0] = 255

            self.viewer.leftMouseButtonPressed.connect(self.handleLeftClick)
            self.viewer.mouseMoved.connect(self.handleMouseMove)

        def run(self):
            # app = QApplication(sys.argv)
            self.viewer.show()
            sys.exit(self.app.exec_())

        def handleLeftClick(self, x, y):
            self.last_pt = (int(x), int(y))
            print 'clicked: {}'.format((x, y))

        def handleMouseMove(self, x, y):
            pt = (int(x), int(y))
            # print 'moved to: {}'.format(pt)
            cv2.line(self.seeds, self.last_pt, pt, 255, 20)
            self.last_pt = pt
            # ok ----
            # im1 = cv2.bitwise_and(self.image, cv2.cvtColor((255 * (self.seeds < 100)).astype(np.uint8), cv2.COLOR_GRAY2RGB))
            # im2 = cv2.bitwise_and(self.red, cv2.cvtColor(self.seeds, cv2.COLOR_GRAY2RGB))
            # self.image_vis = cv2.addWeighted(im1, 1, im2, 1, 0)
            # ----
            self.image_vis = self.image.copy()
            self.image_vis[np.nonzero(self.seeds)] = [255, 0, 0]
            self.qimage = QImage(self.image_vis.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888)
            self.viewer.setImage(self.qimage)


    drawer = Drawer()
    drawer.run()
    # Create the application.
    # app = QApplication(sys.argv)
    #
    # # Create image viewer and load an image file to display.
    # viewer = ImageViewerQt()
    # # viewer.loadImageFromFile()  # Pops up file dialog.
    # image = QImage('data/icons/kky.png')
    # viewer.setImage(image)
    # seeds = np.zeros((image.height(), image.width))
    #
    # # Handle left mouse clicks with custom slot.
    # viewer.leftMouseButtonPressed.connect(handleLeftClick)
    #
    # # Show viewer and run application.
    # viewer.show()
    # sys.exit(app.exec_())