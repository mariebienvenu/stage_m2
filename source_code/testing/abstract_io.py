
import numpy as np

from app.AbstractIO import AbstractIO

def maker():
    return {"test":0, "new":1}

class TestIO(AbstractIO):

    @property
    def config_filename(self):
        return self.directory + "test.json"

    def process(self, force=False):
        self.is_processed = True

    def __repr__(self):
        return super(TestIO, self).__repr__().replace("AbstractIO","TestIO")


directory = "C:/Users/Marie Bienvenu/stage_m2/afac/"
obj = TestIO(directory, verbose=3)
obj.finalize_init(maker)
print(obj)