# Contributing to Captum

We want to make contributing to Captum as easy and transparent as possible.


## Development installation

To get the development installation with all the necessary dependencies for
linting, testing, and building the documentation, run the following:
```bash
git clone https://github.com/pytorch/captum.git
cd captum
pip install -e .[dev]
```


## Our Development Process

#### Code Style

Captum uses [black](https://github.com/ambv/black) and  [flake8](https://github.com/PyCQA/flake8) to
enforce a common code style across the code base. black and flake8 are installed easily via
pip using `pip install black flake8`, and run locally by calling
```bash
black .
flake8 .
```
from the repository root. No additional configuration should be needed (see the
[black documentation](https://black.readthedocs.io/en/stable/installation_and_usage.html#usage)
for advanced usage).

Captum also uses [isort](https://github.com/timothycrosley/isort) to sort imports 
alphabetically and separate into sections. isort is installed easily via
pip using `pip install isort`, and run locally by calling
```bash
isort
```
from the repository root. Configuration for isort is located in .isort.cfg.

We feel strongly that having a consistent code style is extremely important, so
CircleCI will fail on your PR if it does not adhere to the black or flake8 formatting style or isort import ordering.


#### Type Hints

Captum is fully typed using python 3.6+
[type hints](https://www.python.org/dev/peps/pep-0484/).
We expect any contributions to also use proper type annotations, and we enforce 
consistency of these in our continuous integration tests. 

To type check your code locally, install [mypy](https://github.com/python/mypy), 
which can be done with pip using `pip install "mypy>=0.760"`
Then run this script from the repository root:
```bash
./scripts/run_mypy.sh
```
Note that we expect mypy to have version 0.760 or higher, and when type checking, use PyTorch 1.4 or 
higher due to fixes to PyTorch type hints available in 1.4. We also use the Literal feature which is 
available only in Python 3.8 or above. If type-checking using a previous version of Python, you will 
need to install the typing-extension package which can be done with pip using `pip install typing-extensions`.

#### Unit Tests

To run the unit tests, you can either use `pytest` (if installed):
```bash
pytest -ra
```
or python's `unittest`:
```bash
python -m unittest
```

To get coverage reports we recommend using the `pytest-cov` plugin:
```bash
pytest -ra --cov=. --cov-report term-missing
```


#### Documentation

Captum's website is also open source, and is part of this very repository (the
code can be found in the [website](/website/) folder).
It is built using [Docusaurus](https://docusaurus.io/), and consists of three
main elements:

1. The documentation in Docusaurus itself (if you know Markdown, you can
   already contribute!). This lives in the [docs](/docs/).
2. The API reference, auto-generated from the docstrings using
   [Sphinx](http://www.sphinx-doc.org), and embedded into the Docusaurus website.
   The sphinx .rst source files for this live in [sphinx/source](/sphinx/source/).
3. The Jupyter notebook tutorials, parsed by `nbconvert`, and embedded into the
   Docusaurus website. These live in [tutorials](/tutorials/).

To build the documentation you will need [Node](https://nodejs.org/en/) >= 8.x
and [Yarn](https://yarnpkg.com/en/) >= 1.5.

The following command will both build the docs and serve the site locally:
```bash
./scripts/build_docs.sh
```

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you have added code that should be tested, add unit tests.
   In other words, add unit tests.
3. If you have changed APIs, update the documentation. Make sure the
   documentation builds.
4. Ensure the test suite passes.
5. Make sure your code passes both `black` and `flake8` formatting checks.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.


## License

By contributing to Captum, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
