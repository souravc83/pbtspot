import random
import numpy
from matplotlib import pyplot

x = [random.gauss(3,1) for _ in range(400)]
y = [random.gauss(4,2) for _ in range(400)]

bins = numpy.linspace(-10, 10, 100)

pyplot.hist(x, bins, alpha=0.75,label='x',color='r',histtype='barstacked')
pyplot.hist(y, bins, alpha=0.75,label='y',color='b',histtype='barstacked')
pyplot.box('on')
#pyplot.axis('off')

#pyplot.legend(loc='upper right')
pyplot.show()