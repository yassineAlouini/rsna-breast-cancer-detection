
from setuptools import find_packages, setup

NAME = "rsna"
VERSION = "0.0.1"
AUTHOR = ""
DESCRIPTION = """The repo for the RSNA breast cancer detection competition."""
EMAIL = ""
URL = ""

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    # Some metadata
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    url=URL,
    license="MIT",
    keywords="kaggle machine-learning computer-vision deep-learning",
)