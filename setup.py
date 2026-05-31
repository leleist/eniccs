from setuptools import setup, find_packages

setup(
    name='eniccs',
    version='0.2.3',
    packages=find_packages(),
    install_requires=[
              'scikit-learn>=1.4',
              'rasterio>=1.4,<1.5',
              'numpy>=1.26,<2',
              'pandas>=2.0',
              'matplotlib>=3.8',
              'scipy>=1.13',
              'scikit-image>=0.22',
              'joblib>=1.3',
          ],
    python_requires='>=3.11',
    description='EnMAP Improved Cloud and Cloud Shadow (EnICCS) masking pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Leander Leist',
    author_email='leander.leist@geo.uni-marburg.de',
    url='https://github.com/leleist/eniccs',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)