"""
This is the script containing SLM display module (HOLOEYE).
Refer to this interface and modify it to match your SLM SDK.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import utils.detect_heds_module_path
import holoeye
from holoeye import slmdisplaysdk


class SLMDisplay:
    def __init__(self):
        self.ErrorCode = slmdisplaysdk.SLMDisplay.ErrorCode
        self.ShowFlags = slmdisplaysdk.SLMDisplay.ShowFlags

        self.displayOptions = self.ShowFlags.PresentAutomatic  # PresentAutomatic == 0 (default)
        self.displayOptions |= self.ShowFlags.PresentFitWithBars

    def connect(self):
        self.slm_device = slmdisplaysdk.SLMDisplay()
        self.slm_device.open()  # For version 2.0.1

    def disconnect(self):
        self.slm_device.release()

    def show_data_from_file(self, filepath):
        error = self.slm_device.showDataFromFile(filepath, self.displayOptions)
        assert error == self.ErrorCode.NoError, self.slm_device.errorString(error)

    def show_data_from_array(self, numpy_array):
        error = self.slm_device.showData(numpy_array)
        assert error == self.ErrorCode.NoError, self.slm_device.errorString(error)
