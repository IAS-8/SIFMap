# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

FROM python:3.11-slim-bullseye

WORKDIR /app

#Data transfer with volumes
 # cd to project folder
 # build with 'podman build -f docker/Dockerfile -t sifmap-image .'
 # run with 'podman run -it -v .:/app sifmap-image'
COPY ./requirements.txt .

RUN apt-get update
RUN apt-get install libexpat1
RUN apt-get install -y libgomp1 gcc

# Install system dependencies required for GDAL
RUN pip install --no-cache-dir -r requirements.txt

#Data transfer by copying
 # cd to project folder
 # build with 'podman build -f docker/Dockerfile -t sifmap-image .'
 # run with 'podman run -it sifmap-image'
 # retrieve data from container by using `podman cp <Container ID>:/path/in/container /path/to/host/directory`.
 # e.g., podman cp <Container ID>:/app/docker/out/mixed_crops_13062021 ./Output-files
#COPY . /app/
#RUN pip install --no-cache-dir -r /app/requirements.txt

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
CMD ["default"]

