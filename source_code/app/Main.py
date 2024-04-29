
import os, json
from typing import List

import app.AbstractIO as absIO
import app.InternalProcess as InternalProcess
import app.VideoIO as VideoIO
import app.SoftwareIO as SoftIO


def default_config():
    return {
        "video extension": ".mp4",
        "video reference filename" : "ref",
        "video target filename" : "target",
        "blender scene filename" : "scene.blend",
        "connexions of interest": [
            {
                "object name":"Ball",
                "channel":"Location Y",
                "video feature":"First derivative of Velocity Y",
            },
        ],
        "edit in place": False,
    }


class Main(absIO.AbstractIO):

    def __init__(self, directory, verbose=0):
        super(Main, self).__init__(directory, verbose)
        self.finalize_init(default_config)

        self.video_ref = VideoIO.VideoIO(directory=directory, video_name=self.video_reference_filename, extension=self.video_extension, verbose=verbose-1)
        self.video_target = VideoIO.VideoIO(directory=directory, video_name=self.video_target_filename, extension=self.video_extension, verbose=verbose-1)
        self.blender_scene = SoftIO.SoftIO(directory=directory, verbose=verbose-1)
        
        self.blender_scene.check_file(self.directory+self.blender_scene_filename)
        self.internals = None


    def __repr__(self):
        return super(Main, self).__repr__().replace("AbstractIO","Main")

    
    @property
    def config_filename(self) -> str:
        return self.directory + "main_config.json"
    
    @property
    def connexions_of_interest(self) -> List[dict]:
        return self.config["connexions of interest"]
    
    @property
    def video_extension(self) -> str:
        return self.config["video extension"]
    
    @property
    def video_reference_filename(self) -> str:
        return self.config["video reference filename"]
    
    @property
    def video_target_filename(self) -> str:
        return self.config["video target filename"]
    
    @property
    def blender_scene_filename(self) -> str:
        return self.config["blender scene filename"]
    
    @property
    def edit_in_place(self) -> bool:
        return self.config["edit in place"]


    def process(self, force=False):
        if self.is_processed and not force: return self.new_anims

        vanim_ref, vanim_target = self.video_ref.to_animation(), self.video_target.to_animation()
        vanim_ref.time_transl(self.blender_scene.start-vanim_ref.time_range[0]-1) # -1 parce que en dérivant on décale de une frame donc si on veut rester sur start...
        vanim_target.time_transl(self.blender_scene.start-vanim_target.time_range[0]-1)
        vanim_ref.update_time_range()
        vanim_target.update_time_range()
        vanim_ref.enrich()
        vanim_target.enrich()
        banims = self.blender_scene.get_animations()
        #for banim in banims:
        #    banim.enrich() # TODO -- will be useful when we automatically decide of connexions based on multi-modal correlations

        self.internals = [
            InternalProcess.InternalProcess(vanim_ref, vanim_target, banim) for banim in banims
        ]

        warps = [[] for _ in self.internals]
        channels = [[] for _ in self.internals]
        for connexion in self.connexions_of_interest:
            obj_name, feature, channel = connexion["object name"], connexion["video feature"], connexion["channel"]
            index = self.blender_scene.object_names.index(obj_name) ## costly
            internal = self.internals[index]
            warps[index].append(internal.make_simplified_warp(feature=feature, verbose=self.verbose-1)) ## TODO add config to chose between dense and sparse warp based on impulsivity of the signal
            channels[index].append(channel)

        self.new_anims = [internal.make_new_anim(channels=channels[i], warps=warps[i]) for i, internal in enumerate(self.internals)]
        self.is_processed = True
        return self.new_anims


    def to_blender(self):
        self.process()
        self.blender_scene.set_animations(self.new_anims, in_place=self.edit_in_place)

