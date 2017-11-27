from pytocl.analysis import DataLogReader
from pickle import Unpickler
import numpy as np
# (self, filepath, state_attributes=None, command_attributes=None)

filename = "drivelogs/drivelog-2017-11-27-16-38-16.pickle"

read = DataLogReader(filename, ['speed_x'], ['brake'])

# # read.array()
with open(filename, "r") as f:
    read2 = Unpickler(f)

# read.rows(Unpickler())
# for i in read.array:
#     print(i)
a = []
for i in read.array:
    a.append(i[1])

print(max(a) * 3.6)
print(read.state_attributes)
