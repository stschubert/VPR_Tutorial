# Use mambaforge as the base image
FROM condaforge/mambaforge

# Copy environment.yml (you may need to modify the path depending on where your file is located)
COPY environment.yml* ./

# Create the conda environment and install additional packages
RUN mamba env create -f environment.yml && \
    mamba install -n vprtutorial autopep8 black yapf bandit flake8 mypy pycodestyle pydocstyle && \
    mamba clean --all --yes
