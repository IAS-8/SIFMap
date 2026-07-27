#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later


# Copy this file and complete it to set up an environment configuration.

REPO_DIR=".../SIFMap"                # translates to /app in config files when running with docker/podman
DATA_DIR="...."                      # translates to /data in config files when running with docker/podman
USER_DIR='...'                       # translates to /user in config files when running with docker/podman
VENV_DIR=".../venv_sifmap"           # absolute path to your sifmap venv. only for docker/run_wo_docker.sh

IMAGE_NAME="sifmap-image"
DOCKERFILE="docker/Dockerfile"
DOCKEREXEC="podman"                  # depending on the available installation
SHM="100g"                           # shared memory available during docker/podman execution

