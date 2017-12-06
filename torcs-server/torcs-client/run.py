#!/usr/bin/env python3

import sys
print(sys.version_info[0])

from pytocl.main import main
from human_driver import MyDriver


if __name__ == '__main__':

    main(MyDriver())
