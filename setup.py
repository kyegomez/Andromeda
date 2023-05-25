from setuptools import setup, find_packages

setup(
  name = 'andromeda_transformer',
  packages = find_packages(exclude=['examples']),
  version = '1.1.3',
  license='MIT',
  description = 'andromeda - Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  url = 'https://github.com/kyegomez/Andromeda',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.6.1',
    'datasets',
    'accelerate',
    'transformers',
    'optimus-prime-transformers',
    'lion_pytorch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)