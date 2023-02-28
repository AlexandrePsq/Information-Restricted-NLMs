import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="irnlm",
    version="0.0.1",
    author="Alexandre Pasquiou",
    author_email="alexandre.pasquiou@inria.fr",
    description="Python code to reproduce the analyses of the paper: `Information-Restricted Neural Language Models Reveal Different Brain Regions' Sensitivity to Semantics, Syntax and Context`",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexandrePsq/Information-Restrited-NLMs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["nilearn", "nibabel", "torch", "scipy", "PyYAML", "matplotlib", "tqdm", "scikit-learn", "pytest"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
