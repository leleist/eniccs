from setuptools import setup, find_packages

setup(
    name='eniccs',
    version='0.11.0-alpha',
    packages=['eniccs'],
    install_requires=[
        'scikit-learn==1.3.2',
        'rasterio==1.3.10',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'matplotlib==3.7.5',
        'jupyter==1.0.0',
        'scipy==1.10.1',
    ],
    python_requires='>=3.8.19',
    description='Improvement of cloud and cloudshadow detection in hyperspectral EnMAP images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # Format of the README file
    author='Leander Leist',
    author_email='science.leist@gmail.com',
    url='https://github.com/leleist/eniccs',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
