<!--
SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
SPDX-FileContributor: Armagan Elibol
SPDX-FileContributor: Christian Hirt
SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Docker Container Overview

In this Markdown file the details to the Docker container will be explained.

## Configuration File
Before running the program, the configuration file must be customized. Which configuration file is used when the program is executed is specified in the last line of the `Dockerfile`. The third parameter of the Entrypoint command specifies the path to the config file in the Docker container. The paths to the data inside the configuration file must be specified for the environment in the container. The working directory of the Docker image is set to `/app`. So when creating the config file all file-paths should start with `/app`. 
The configuration file includes a parameter called `manual_outlier_mode`. When set to true, it requires the user to manually finalize the outlier removal process by entering 0 or 1 in the terminal. To use this option, the Docker image must be run with the `-it` flag.
The `n_processes` parameter in the configuration file determines the number of jobs to be created. If this parameter is set too high, it can cause excessive memory consumption within the Docker container, potentially leading to the program's termination without generating an error message. This usually happens when visualizing the results.

## Adding Files to the Container
To run the Program the container needs both the Python files as well as the data located in example\_data.
You can either Copy the files into the container by using the `COPY` command in the Dockerfile or use a volume with the option `-v` when running the dockerfile. 

### Using the `COPY` Command 
The `COPY` command in the `Dockerfile` copies all data from a folder in the host to a folder in the container environment when building the Image. From there, the files can be used when running the program.
Files added to the `.dockerignore` are not copied into the container.  The previously created files inside the output folder should not be copied into the container.
When using the `COPY` command, only files and directories located in the same directory as the Dockerfile or subdirectories within it can be copied into the container. This is because the Docker build context is limited to the directory where the Dockerfile is located. For data outside the folder with the Dockerfile, a volume should be used. 

If the output directory is not a volume, but was copied into the Docker container, the output data created when running the program is only saved in the container and not saved on the executing device. To transfer the data, `podman cp <Container ID>:/path/in/container /path/to/host/directory` can be used. E.g. `podman cp <Container ID>:/app/docker/out/mixed_crops_13062021 ./Output-files`. The Container ID can be determined with the command `podman ps -a`.

### Using a Volume
A volume allows data to be shared between the container and the host system at runtime. Unlike the COPY command, which transfers files into the container during the image build, a volume mounts a host directory to a container directory. 
If the output directory is located within a volume, all files created are transferred to the system outside the container.
A volume can be specified by including the `-v` option when executing the built image. `podman run -v /path/to/data:/path/in/container`.
It should be noted that the requirements file must already be copied into the container when building it with the Dockerfile. 
To avoid data being written into a volume use a read only volume by appending `:ro` to the `-v` parameter. E.g. `podman run -v /path/to/data:/app/data:ro`. Note that in this case the output folder must not be part of the volume.

## Building and Running the Image

* Change the current working-directory to the folder with the Dockerfile.

* Build the Image using `podman build -t <image-name> .`

* Run the image and specify optional parameter as described above. `podman run (-it) (-v /path/to/host:/app) <image-name>`

* Optional: if the data was transferred with the COPY command, it can be transferred to the host system with `podman cp <Container ID>:/path/in/container /path/to/host/directory`

## Base Image Details
The Docker file uses the base image `python:3.11-slim-bullseye`.
This image does not contain the common Debian packages from the default python image but only the minimal Debian packages needed to run python. This reduces the size of the final image. For the currently used packages this is sufficient however to install some packages with `pip install` in the future it may be necessary to install the required Debian packages.
For more details search [here](https://hub.docker.com/_/python) under `Image Variants`, `python:<version>-slim`.
The Image Variant alpine is even smaller. However there are no pre-built wheels available for the package `scikit-image` for this variant. It would be possible to build the package from source but this requires more work and is likely to face more issues. The advantage of alpine compared to slim is minimal in terms of image pull time and additional disk space thus slim is used.

## Python Package Installation
The required python packages from the `requirements.txt` file are installed with `pip install`.
Additionally the option `--no-cache-dir` is used to avoid using a cache when installing the packages. This is a [good practice](https://docs.datadoghq.com/code_analysis/static_analysis_rules/docker-best-practices/pip-no-cache/) to prevent using potentially outdated packages from the cache, which might have security vulnerabilities or bugs.

### Using `opencv-python-headless`
The required packages are specified in the `requirements.txt` file. The `opencv-python-headless` package is used here instead of `opencv-python`. 
The reason for this is that `opencv-python` has some additional dependencies that need to be installed.
To avoid this, the headless alternative is used, which does not include any GUI functionality and therefore has fewer dependencies and is also smaller than the complete variant. For more details search [here](https://pypi.org/project/opencv-python-headless/) under `Installation and Usage`, `3 Select the correct package `, `b Packages for server`. 
Additionally on Stackoverflow a user suggested this for CPU-based opencv activities. Perhaps using the package will cause problems when calculating with the GPU.






