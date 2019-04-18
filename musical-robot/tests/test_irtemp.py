import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irtemp

def test_centikelvin_to_celcius():
    cels = irtemp.centikelvin_to_celsius(100000)
    assert isinstance(cels, float),'Output is not a float'
