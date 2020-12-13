# ---
# @Software: PyCharm
# @File: trial_store.py
# @Author: 姚君彦
# @Time: 2020/12/12,15:13
# ---
import pickle

filename = 'ser'

x = 'a'

y = '100'

z = '100'

ma = [x,y,z]

with open(filename, 'wb') as f:
    pickle.dump(ma, f)

    # pickle.dump(y, f)
    #
    # pickle.dump(z, f)

with open(filename, 'rb')as f:

    a = pickle.load(f)

    print("\n")
    for i in a:
        print(i)
    print("\n")

