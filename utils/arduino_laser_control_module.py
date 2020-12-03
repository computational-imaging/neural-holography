"""
This is the script containing Arduino controller module using pyfirmata.
If you don't want to automate the laser control using python script, just turn off laser_arduino option of PhysicalProp.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""


try:
    from pyfirmata import Arduino, util
except (ImportError, ModuleNotFoundError):
    import pip
    pip.main(['install', 'pyfirmata'])
    from pyfirmata import Arduino, util

import time


class ArduinoLaserControl:

    color2index = {'r': 0, 'g': 1, 'b': 2}
    ind2color = ['r', 'g', 'b']

    def __init__(self, port='/dev/cu.usbmodem14301', pins=None):
        """

        :param port: the port of your Arduino.
                     You may find the port at
                     your Arduino program -> Tool tab -> port         (Mac)
                     Device manager -> Ports (COM & LPT)              (Windows)

        :param pins: an array of Arduino d pins with PWM
        """
        self.board = Arduino(port)
        self.pinNums = [6, 10, 11] if pins is None else pins
        self.pins = {}
        self.default_pin = self.board.get_pin(f'd:3:p')

        for c in self.color2index:
            self.pins[c] = self.board.get_pin(f'd:{self.pinNums[self.color2index[c]]}:p')

    def setPins(self, pins):
        """
        set output pins of arduinos, for control

        :param pins: an array of RGB pin numbers - PWM capable - at your Arduino Uno (e.g. [9, 10, 11])
        """
        self.pinNums = pins
        for c in self.color2index:
            self.pins[c] = self.board.get_pin(f'd:{self.pinNums[self.color2index[c]]}:p')

    def setValue(self, colors, values):
        """

        :param colors: an array or chars ('r' or 'g' or 'b'), single char (e.g. 'r') is acceptable
        :param values: an array of normalized values (corresponds to the percent in the control box)
                       for each color.
                       e.g. [0.4 0.1 1]
                       if you want identical values for all colors just put a scalar
        """

        # check whether parameter is scalar or array
        if isinstance(colors, list) is False:
            if len(colors) > 1:
                colors = colors[0]
            colors = [colors]

        numColors = len(colors)

        if not isinstance(values, list):
            values = [values] * numColors

        if len(values) != len(colors):
            print("  - LASER CONTROL : Please put the same number of values to 'colors' and 'values' ")
            return

        # turn on each color
        for i in range(numColors):

            # Detect color
            if colors[i] in self.color2index:
                pin = self.pins[colors[i]]
            else:
                # colors must be 'r' or(and) 'g' or(and) 'b'
                print("  - LASER CONTROL: Wrong colors for 'setValue' method, it must be 'r' or(and) 'g' or(and) 'b'")
                return

            print(f'  - V[{colors[i]}] from Arduino : {values[i]:.3f}V\n')
            pin.write(values[i])

    def switch_control_box(self, channel):
        """
        switch color of laser through control box
        with D-Sub 9pin, but note that it uses only 4-bit encoding.

        R: 1100
        G: 1010
        B: 1001

        :param channel: integer, channel to switch (Red:0, Green:1, Blue:2)
        """
        self.default_pin.write(1.0)
        time.sleep(10.0)

        if channel in [0, 1, 2]:
            self.pins[self.ind2color[channel]].write(1.0)
            for c in [0, 1, 2]:
                if c != channel:
                    self.pins[self.ind2color[c]].write(0.0)
        else:
            print('turning off')
            for c in [0, 1, 2]:
                self.pins[self.ind2color[c]].write(0.0)
            time.sleep(10.0)
            self.default_pin.write(1.0)

    def turnOffAll(self):
        """
        Feed 0000 to control box

        :return:
        """
        for c in self.color2index:
            pin = self.pins[c]
            pin.write(0)
        self.switch_control_box(3)

        print('  - Turned off')

    def disconnect(self):
        self.turnOffAll()
