import io
import os
import sys

from setuptools import setup

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6.0 is not supported")

DESCRIPTION = "Chainer PrefetchMultiprocessIterator"
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# load __version__
exec(open(os.path.join(here, "prefetch_multiprocess_iterator", "_version.py")).read())

setup(
    name="chainer_prefetch_multiprocess_iterator",
    version=__version__,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kazuhiro Serizawa",
    author_email="nserihiro@gmail.com",
    url="https://github.com/serihiro/chainer_prefetch_multiprocess_iterator",
    license="MIT",
    packages=["prefetch_multiprocess_iterator"],
    install_requires=["chainer>=6.4.0"],
)
