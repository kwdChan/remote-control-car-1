import cv2
class Camera:
    def __init__(self, camera_idx=0, name='camera'):
        self.cap = cv2.VideoCapture(camera_idx)
        self.name = name

    def sample(self, logger=None):
        """
        the code is blocking. it waits for the frame
        
        """
        ret, img = self.cap.read()

        if not logger is None:
            logger.log(self.name, img)
        return img