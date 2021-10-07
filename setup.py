from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='edotools',
      version='0.1.2',
      description='AutoML',
      packages=['edotools'],
      author_email='edvard88@inbox.ru',
      long_description=long_description,
      long_description_content_type='text/markdown',
      zip_safe=False)