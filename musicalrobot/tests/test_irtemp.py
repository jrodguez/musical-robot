


def test_centikelvin_to_celcius():
    '''Test: Converts given centikelvin value to Celsius'''
    cels = irtemp.centikelvin_to_celsius(100000)
    assert isinstance(cels, float),'Output is not a float'
    return

def test_to_fahrenheit():
    '''Test: Converts given centikelvin reading to fahrenheit'''
    fahr = irtemp.to_fahrenheit(100000)
    assert isinstance(fahr, float), 'Output is not a float'
    return

def test_to_temperature():
    '''Test: Converts given centikelvin value to both fahrenheit and celcius'''
    cels, fahr = irtemp.to_temperature(100000)
    assert isinstance(fahr, float), 'Output is not a float'
    assert isinstance(cels, float),'Output is not a float'
    return
