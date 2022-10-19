import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cyy_ml_if",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/ML_influence_function_family",
    packages=[
        "cyy_ml_if",
        "cyy_ml_if/hydra",
        "cyy_ml_if/lean_hydra",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
