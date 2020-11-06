import numpy as np
import cv2
from os import path


class MaskPainter():
    def __init__(self, image_path, operation):
        """
        image_path: the path of the original image
        operation: 0 means increasing the brightness
        """
        self.image = cv2.imread(image_path)
        print(image_path)
        self.image_path = image_path
        self.image_copy = self.image.copy()

        self.mask_name = "./figs/mask.png"
        # matrix g, which is the image after the user's rough edit
        self.mask = np.zeros(self.image.shape[:2])
        self.mask_copy = self.mask.copy()
        self.size = 1  # size of brush
        self.to_draw = False

        self.window_name = "Draw a mask. s:save; r:reset; q:quit"

    def _paint_mask_handler(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_draw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_draw:
                cv2.rectangle(self.image, (x-self.size, y-self.size),
                              (x+self.size, y+self.size),
                              (0, 255, 0), -1)
                cv2.rectangle(self.mask, (x-self.size, y-self.size),
                              (x+self.size, y+self.size),
                              (255, 255), -1)
                cv2.imshow(self.window_name, self.image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.to_draw = False

    def paint_mask(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,
                             self._paint_mask_handler)

        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.image = self.image_copy.copy()
                self.mask = self.mask_copy.copy()

            elif key == ord("s"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)
        # self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        print("-----The shape of mask is:", self.mask.shape)

        cv2.imwrite(self.mask_name, self.mask)

        # close all open windows
        cv2.destroyAllWindows()
        return self.mask_name
