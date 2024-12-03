from setuptools import find_packages, setup

setup(
    name="torchFastText",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "src"},
    include_package_data=True,  # Ensure additional files like README.md are included
    install_requires=["pytorch_lightning", "captum", "unidecode", "nltk", "scikit-learn"],
)
