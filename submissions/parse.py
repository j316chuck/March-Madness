

f = open('seed.csv', 'rb')
w = open('seed2.csv', 'w')
w.write('id,pred\n')
for line in f.readlines():
	for ln in line.split():
		y, m, a = ln.split(',')
		w.write(y + ',' + m + '\n')	
