from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rshap",
    version="0.1.0",
    author="Prof Amanda S Barnard",
    author_email="amaxiom@gmail.com",
    description="Residual Decomposition Symmetric — instance-level influence analysis via Shapley-style residual attribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amaxiom/RSHAP",
    py_modules=["RSHAP"],
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "shapley", "explainability", "instance analysis",
        "residual decomposition", "machine learning", "interpretability"
    ],
)
