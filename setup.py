from setuptools import setup, find_packages

setup(
    name='eniccs',
    version='0.1.0',  # Changed from '0.19.0-alpha'
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.7.0',
        'rasterio>=1.4.3',
        'numpy>=2.3.0',
        'pandas>=2.3.0',
        'matplotlib>=3.10.3',
        'scipy>=1.15.3',
        'scikit-image>=0.25.0',
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