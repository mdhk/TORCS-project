import pickle

# print(pickle.load( open( "drivelog-2017-11-17-10-04-33.pickle", "rb" ) ))

with open( "drivelogs/drivelog-2017-11-17-10-04-33.pickle", "rb" ) as handle:
    a = []

    a.append(pickle.load(handle))
    print(a)
