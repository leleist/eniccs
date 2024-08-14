# EnICCS

**Project Name** is a brief description of what your project does and its main features.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install **Project Name**, you can use pip:

```bash
pip install project-name
```

For the latest fixes, you can clone the repository and install it:
```bash
git clone https://github.com/username/repository.git
cd repository
pip install -e .
```

## Usage
To use **EnICCS**, you can import it and run the main wrapper function in default configuration.  
Just provide the directory path of the EnMAP data

```python
import eniccs

dir_path = r"path/to/your/EnMAP/TIFFS"
eniccs(dir_path)
```
new cloud and cloudshadow masks will be saved to "dir_path"

## Examples
Here is an example of how to use **EnICCS**:

```python



