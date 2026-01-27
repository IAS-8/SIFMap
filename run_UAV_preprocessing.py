# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from run_utils import run_preprocessing, read_config


def main():
    if len(sys.argv) > 1:
        config = sys.argv[1]
        print(f"Running: {config}")
    else:
        print("You need to provide a config file.")

    params = read_config(config)
    run_preprocessing(**params)


if __name__ == '__main__':
    main()
