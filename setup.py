# SET UP FILE:
# in order to run in notebooks as an import musicalrobot
# 1. Git clone the repository to a local computer
# 2. go to the outermost musical-robot folder
# 3. use "pip install . "
# 4. import packages into a jupyter notebook using "from musicalrobot import xxxxxx"

from setuptools import setup

setup(name = 'musical-robot',
    version = '0.1',
    packages = ['musicalrobot'],
    url = 'https://github.com/pozzocapstone/musical-robot',
    license = 'MIT',
    author = 'Shrilakshmi Bonageri, Jaime Rodriguez, Sage Scheiwiller',
    short_description = 'Melting temperature determination using IR bolometry',
    long_description = open('README.MD').read(),
    zip_safe = False
)