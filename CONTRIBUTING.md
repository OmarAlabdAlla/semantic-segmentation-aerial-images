# Contributing to Semantic Segmentation on Potsdam

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/semantic-segmentation-potsdam.git
   cd semantic-segmentation-potsdam
   ```
3. **Set up the development environment** (see below)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment (OS, Python version, GPU, etc.)
- Error messages or logs if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- A clear description of the proposed feature
- The motivation or use case
- Any relevant examples or references

### Types of Contributions Needed

- **Bug fixes**: Fix issues in existing code
- **New models**: Implement additional segmentation architectures
- **Documentation**: Improve README, add tutorials, or clarify existing docs
- **Tests**: Add unit tests or integration tests
- **Optimization**: Improve training speed or reduce memory usage
- **Visualizations**: Create better result visualizations

---

## Development Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install black isort flake8 pytest
```

### Dataset Setup

1. Download the Potsdam dataset from [ISPRS](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
2. Update the `data_dir` path in the notebooks

---

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and reasonably sized

### Code Formatting

We recommend using these tools:

```bash
# Format code
black your_file.py

# Sort imports
isort your_file.py

# Check for issues
flake8 your_file.py
```

### Jupyter Notebooks

- Clear all outputs before committing (reduces file size)
- Use markdown cells to explain complex sections
- Keep cells reasonably sized
- Include comments for non-obvious code

### Commit Messages

Write clear, concise commit messages:

```
# Good examples:
feat: Add SegFormer-B2 model implementation
fix: Correct class weight calculation for edge cases
docs: Update installation instructions for Windows
refactor: Simplify data augmentation pipeline

# Format:
<type>: <short description>

<optional longer description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

## Submitting Changes

### Pull Request Process

1. **Update documentation** if your changes affect usage
2. **Test your changes** thoroughly
3. **Clear notebook outputs** before committing
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** on GitHub with:
   - A clear title and description
   - Reference to any related issues
   - Screenshots/results if applicable

### Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review of the code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated if needed
- [ ] No new warnings introduced
- [ ] Notebook outputs cleared

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

---

## Questions?

If you have questions, feel free to:

- Open an issue with the `question` label
- Reach out to the maintainer directly

Thank you for contributing!
