import LoReGro_App
import numpy as np

def run( im, energy = np.empty(0) ):

    help_message = '''
      USAGE: watershed.py [<image>]

      Use keys 1 - 2 to switch marker color
      SPACE - update segmentation
      r     - reset
      a     - switch autoupdate
      ENTER - run segmentation
      ESC   - exit

    '''
    print help_message
    LoReGro_App.LoReGro_App( im, energy ).run()
