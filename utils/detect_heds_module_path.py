# -*- coding: utf-8 -*-

#--------------------------------------------------------------------#
#                                                                    #
# Copyright (C) 2020 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# 'HOLOEYE SLM Display SDK Standard License v1.0' license agreement. #
#                                                                    #
#--------------------------------------------------------------------#


# Please import this file in your scripts before actually importing the HOLOEYE SLM Display SDK,
# i. e. copy this file to your project and use this code in your scripts:
#
# import detect_heds_module_path
# import holoeye
#
#
# Another option is to copy the holoeye module directory into your project and import by only using
# import holoeye
# This way, code completion etc. might work better.


import os, sys
from platform import system

# Import the SLM Display SDK:
HEDSModulePath = os.getenv('HEDS_2_PYTHON_MODULES', '')

if HEDSModulePath == '':
    sdklocal = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'holoeye', 'slmdisplaysdk', '__init__.py'))
    if os.path.isfile(sdklocal):
        HEDSModulePath = os.path.dirname(os.path.dirname(os.path.dirname(sdklocal)))
    else:
        sdklocal = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                                'sdk', 'holoeye', 'slmdisplaysdk', '__init__.py'))
        if os.path.isfile(sdklocal):
            HEDSModulePath = os.path.dirname(os.path.dirname(os.path.dirname(sdklocal)))

if HEDSModulePath == '':
    if system() == 'Windows':
        print('\033[91m'
              '\nError: Could not find HOLOEYE SLM Display SDK installation path from environment variable. '
              '\n\nPlease relogin your Windows user account and try again. '
              '\nIf that does not help, please reinstall the SDK and then relogin your user account and try again. '
              '\nA simple restart of the computer might fix the problem, too.'
              '\033[0m')
    else:
        print('\033[91m'
              '\nError: Could not detect HOLOEYE SLM Display SDK installation path. '
              '\n\nPlease make sure it is present within the same folder or in "../../sdk".'
              '\033[0m')

    sys.exit(1)

sys.path.append(HEDSModulePath)
