from setuptools import find_packages, setup

setup(
    name="pacotesoma",
    version="0.1.0",
    description="Pacote simples para somar dois números (demonstração)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Seu Nome",
    author_email="seu.email@example.com",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
