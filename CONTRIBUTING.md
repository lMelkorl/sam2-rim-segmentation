# Contributing to SAM2 Rim Segmentation

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project.

## Report bugs using GitHub's [issues](https://github.com/lmelkorl/sam2-rim-segmentation/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/lmelkorl/sam2-rim-segmentation/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. Clone your fork of the repository:
```bash
git clone https://github.com/lmelkorl/sam2-rim-segmentation.git
cd sam2-rim-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
cd sam2 && pip install -e . && cd ..
```

3. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

## Coding Style

- Use meaningful variable and function names
- Add comments for complex logic
- Follow PEP 8 for Python code
- Use type hints where appropriate

## Testing

- Test your changes with different types of vehicle images
- Ensure the web interface works properly
- Verify that mask generation produces quality results

## Documentation

- Update README.md if you add new features
- Add docstrings to new functions
- Update this CONTRIBUTING.md if the development process changes

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## Questions?

Feel free to open an issue for any questions about contributing! 