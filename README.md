<div align="center">
<h1> BeamZ </h1>

[![PyPI version](https://badge.fury.io/py/beamz.svg)](https://badge.fury.io/py/beamz)
[![Last Commit](https://img.shields.io/github/last-commit/QuentinWach/beamz)](https://github.com/QuentinWach/beamz/commits/main)

Create inverse designs for your photonic devices with ease and efficiency.

[Homepage]() •
[Documentation](#documentation) •
[Examples](#examples) •
[Development](#development) •
[Citation](#cite)
</div>


```bash
pip install beamz
```

## Features
+ **Fast FDTD** with native GPU-acceleration (but CPU multi-threading option).
+ **Topology optimization** using the adjoint method & auto-differentiation.
+ **Simple API** for quick .gds import and design of structures.
+ **Material Library** including dispersive and non-dispersive material models.


## Examples
Coming soon...


## Documentation
Coming soon...

We prefer not to clutter the code with lengthy explanations but to keep the docstrings concise, covering the essentials like purpose, parameters, and return values. We expand on the details, usage examples, and tutorials within your dedicated docs/ folder.


## Development

_BeamZ_ is currently in early developement and not feature complete or optimized for speed. That means any contributions you make now will have a major impact!

To get started, install _BeamZ_ for developement:
```bash
git clone https://github.com/QuentinWach/beamz.git
cd beamz
pip install -e .
```


### Priorities

1. _BeamZ_ optimized for the GPU for speed but also provides a multi-threaded CPU backend so you can **run it on any device** and expect it to be **fast** while keeping it **local** and your **IP secure**!

2. _BeamZ_ is a **community project**. That means making it as easy to use and as easy to contribute to this project are priorities rather than optimizing the software for speed or adding a long list of features few people will ever use and just overcomplicate developement. An extensive documentation, examples, and community building are on top of the list but keeping the code compact is more important than spamming comments for documentation purposes all around it.



### Contributing Through Git
You can see open issues and feauture requests on GitHub. To contribute towards the developement follow the genral steps below:

1. Fork the repository on GitHub.
2. Clone your fork locally and install the dependencies:
   ```bash
   git clone https://github.com/QuentinWach/beamz.git
   cd beamz
   pip install -e .
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
5. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request (PR) on GitHub from your branch.

In general, making commits can be easily done in the VSCode GUI but we recommend actually using git commands in the terminal for pushing changes and creating new branches. 


### Test-driven Development (TDD)

We are developing _BeamZ_ with a test-first approach. For any new feature, you first write tests that will check if the feature is working, then you write the feature, running the tests to see when it is indeed working:

1. **Write a Failing Test**
   - Create a new test file in the `tests/` directory.
   - Write a test that describes the expected behavior of your feature.
   - The test should fail initially since the feature doesn't exist yet.
   ```python
   def test_new_feature():
       # Arrange
       input_data = ...
       expected_output = ...
       
       # Act
       result = your_new_feature(input_data)
       
       # Assert
       assert result == expected_output
   ```

2. **Write the Feature**
   - Implement the feature to make the test pass.
   - Keep the implementation simple and focused on passing the test.
   - Don't add functionality that isn't tested.

3. **Refactor**
   - Clean up the code while keeping all tests passing.
   - Improve readability and maintainability.
   - Remove any duplication.

4. **Repeat**
   - Add more test cases to cover edge cases.
   - Enhance the feature based on new requirements.
   - Keep the cycle going until the feature is complete.

#### Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_your_feature.py

# Run tests with coverage report
pytest --cov=beamz tests/
```

#### Best Practices

- Write tests that are independent of each other.
- Use meaningful test names that describe the expected behavior.
- Follow the Arrange-Act-Assert pattern in your tests.
- Keep tests simple and focused on one behavior.
- Use fixtures for common setup code.
- Mock external dependencies when appropriate.


## Citation

If you use BeamZ in your research, please cite it using the following BibTeX entry:

```bibtex
@software{beamz2025,
  author = {Wach, Quentin},
  title = {BeamZ: A Fast and Efficient Tool for Photonic Device Inverse Design},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/QuentinWach/beamz},
  version = {0.0.1}
}
```

You can also cite the specific version you used by including the DOI from Zenodo (coming soon).