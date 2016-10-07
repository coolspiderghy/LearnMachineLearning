a = [1, 2, 3, 4, 5, 6]
print iter(a)
for i in iter(a):
    print i
print zip(*([iter(a)] * 3))