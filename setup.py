from setuptools import setup, find_packages

setup(
    name='eniccs',
    version='0.19.0-alpha',
    packages=['eniccs'],
    install_requires=[
        'scikit-learn>=1.7.0',
        'rasterio>=1.4.3',
        'numpy>=2.3.0',
        'pandas>=2.3.0',
        'matplotlib>=3.10.3',
        'jupyter>=1.1.1',
        'scipy>= 1.15.3',
        'scikit-image >=0.25.0',
    ],
    python_requires='>=3.11',
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
        'Programming Language :: Python :: 3.11',
    ],
)
