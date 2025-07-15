import setuptools

setuptools.setup(
    name = "bailstudy",
    version = "0.0.1",
    author = "Phylliida",
    author_email = "phylliidadev@gmail.com",
    description = "Studying bail behavior of LLMs",
    url = "https://github.com/Phylliida/BailStudy.git",
    project_urls = {
        "Bug Tracker": "https://github.com/Phylliida/BailStudy/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    python_requires = ">=3.6",
    install_requires = ["vllm[tensorizer]", "flashinfer-python","pandas", "numpy", "torch", "ujson", "setuptools", "pyarrow", "markdownify", "pytz", "huggingface-hub", "langchain", "transformers", "safetytooling @ https://github.com/safety-research/safety-tooling.git"]
)
