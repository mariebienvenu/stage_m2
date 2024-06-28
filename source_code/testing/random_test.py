
import numpy as np

# Advanced slicing

arr = np.array([[i*5+j for j in range(5)] for i in range(5)])
print(arr)
content = arr[(1,2),(3,4)]
other_content= arr[np.ix_((1,2),(3,4))]
print(content, other_content)

## Using axiliary functions as variables and attributes in class

class Foo:

    def __init__(self, foo):
        self.foo = foo
        self.aux = self.process()

    def process(self):
        truc = 2
        def aux(p):
            return p+truc
        return aux
    
    def make(self):
        self.foo = self.aux(self.foo)

obj = Foo(2)
print(obj.foo)
obj.make()
print(obj.foo)