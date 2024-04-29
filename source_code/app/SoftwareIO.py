
import os
from typing import List

import app.blender_utils as b_utils
import app.Animation as Animation
import app.AbstractIO as AbstractIO
# bientôt: import app.maya_utils as maya_utils

def default_config():
    return {
        "time range": {
            "start": 1,
            "stop": 100,
        },
        "object names": [
            "Ball",
        ],
    }


class SoftIO(AbstractIO.AbstractIO):

    def __init__(self, directory, verbose=0):
        super(SoftIO, self).__init__(directory, verbose)
        self.finalize_init(default_config)


    def __repr__(self):
        return super(SoftIO, self).__repr__().replace("AbstractIO","SoftIO")

    
    @property
    def config_filename(self):
        return self.directory + "scene_config.json"
    
    @property
    def scene_filename(self) -> str:
        return b_utils.get_filename()
    
    @property
    def time_range(self) -> dict:
        return self.config["time range"] 
    
    @property
    def start(self) -> int:
        return self.time_range["start"]

    @property
    def stop(self) -> int:
        return self.time_range["stop"]
    
    @property
    def object_names(self) -> List[str]:
        return self.config["object names"]


    def process(self, force=False, backup=True):
        if self.is_processed and not force : return
        self.from_software()
        if backup: 
            self.original_anims = [
                b_utils.get_animation(obj_name) for obj_name in self.object_names
            ]
        self.is_processed = True


    def get_animations(self):
        self.process()
        return self._animations


    def set_animations(self, animations:List[Animation.Animation], in_place=False):
        self._animations = animations
        self.to_software(in_place=in_place)
    

    def check(self):
        for anim in self._animations:
            anim.check()

    
    def check_file(self, filename):
        b_utils.check_filename(filename)


    def to_software(self, in_place=False):
        for (obj_name, animation) in zip(self.object_names, self._animations):
            target_obj = obj_name if in_place else obj_name+"_edited"
            b_utils.set_animation(target_obj, animation)
        self.process(force=True)


    def from_software(self):
        self._animations = [
            b_utils.get_animation(obj_name) for obj_name in self.object_names
        ]
        for anim in self._animations:
            anim.crop(self.start, self.stop)
        self.check()