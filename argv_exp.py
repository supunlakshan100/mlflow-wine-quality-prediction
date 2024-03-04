import sys

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

print("alpha: %.2f" % alpha, "l1_ratio: %.2f" % l1_ratio)

'''import numpy as np
 
print("My numpy version is: ", np.__version__)'''