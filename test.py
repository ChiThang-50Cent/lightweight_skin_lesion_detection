a = 'abc'

t = a + '{epoch:02d}'
t = t.format(epoch=10)

print(t)