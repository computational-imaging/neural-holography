"""
This is the script containing Camera module (PyCapture).
Refer to this interface and modify it to match your camera SDK

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import PyCapture2
import cv2
import numpy as np
import time
import utils.utils as utils


def callback_captured(image):
    print(image.getData())


class CameraCapture:
    def __init__(self):
        self.bus = PyCapture2.BusManager()
        num_cams = self.bus.getNumOfCameras()
        if not num_cams:
            exit()

    def connect(self, i):
        uid = self.bus.getCameraFromIndex(i)
        self.camera_device = PyCapture2.Camera()
        self.camera_device.connect(uid)
        self.toggle_embedded_timestamp(True)

    def disconnect(self):
        self.toggle_embedded_timestamp(False)
        self.camera_device.disconnect()

    def toggle_embedded_timestamp(self, enable_timestamp):
        embedded_info = self.camera_device.getEmbeddedImageInfo()
        if embedded_info.available.timestamp:
            self.camera_device.setEmbeddedImageInfo(timestamp=enable_timestamp)

    def grab_images(self, num_images_to_grab=1):
        """
        Retrieve the camera buffer and returns a list of grabbed images.

        :param num_images_to_grab: integer, default 1
        :return: a list of numpy 2d color images from the camera buffer.
        """
        self.camera_device.startCapture()

        img_list = []
        for i in range(num_images_to_grab):
            try:
                img = self.camera_device.retrieveBuffer()
            except PyCapture2.Fc2error as fc2Err:
                continue

            imgData = img.getData()

            # when using raw8 from the PG sensor
            # cv_image = np.array(img.getData(), dtype="uint8").reshape((img.getRows(), img.getCols()))

            # when using raw16 from the PG sensor - concat 2 8bits in a row
            imgData.dtype = np.uint16
            imgData = imgData.reshape(img.getRows(), img.getCols())
            offset = 64  # offset that inherently exist.
            imgData = imgData - offset

            color_cv_image = cv2.cvtColor(imgData, cv2.COLOR_BAYER_RG2BGR)
            color_cv_image = utils.im2float(color_cv_image)
            img_list.append(color_cv_image.copy())

        self.camera_device.stopCapture()
        return img_list

    def start_capture(self):
        # these two were previously inside the grab_images func, and can be clarified outside the loop
        self.camera_device.startCapture()

    def stop_capture(self):
        self.camera_device.stopCapture()
