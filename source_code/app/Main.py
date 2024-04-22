
import os, json

import app.InternalProcess as InternalProcess
import app.VideoIO as VideoIO
import app.blender_utils as b_utils


def default_config():
    return {
        "connexions of interest": [
            {
                "feature":"First derivative of Velocity Y",
                "channel":"Location Y"
            },
        ],
    }


class Main:

    def __init__(self, directory, blender_object_name, reference_video_filename, target_video_filename, extension='.mp4',  verbose=0):

        self.video_ref = VideoIO.VideoIO(directory=directory, video_name=reference_video_filename, extension=extension, verbose=verbose-1)
        self.video_target = VideoIO.VideoIO(directory=directory, video_name=target_video_filename, extension=extension, verbose=verbose-1)
        self.directory = directory
        self.blender_object_name = blender_object_name

        self.config = None
        self.config_loaded = False
        
        self.internal = None
        self.is_processed = False

        try:
            self.load_config()
            self.complete_config()
        except OSError:
            print("Did not find config file ; reverting to default config.") if verbose>0 else None
            self.make_default_config()

        assert self.config_loaded, "Error when initializing Main object."


    def make_default_config(self):
        self.config = default_config()
        self.config_loaded = True

    
    def complete_config(self):
        default = default_config()
        for key, value in default.items():
            if key not in self.config:
                self.config[key] = value

    def save_config(self):
        assert self.config_loaded, "Cannot save config when no config is loaded."
        with open(self.directory+"main_config.json", "w") as outfile:
            json.dump(self.config, outfile)


    def load_config(self, force=False):
        if self.config_loaded and not force:
            return
        if not os.path.exists(self.directory+"main_config.json"):
            raise OSError("No config file.")
        with open(self.directory+"main_config.json", 'r') as openfile:
            self.config = json.load(openfile)
        self.config_loaded = True


    @property
    def connexions_of_interest(self):
        return [(d['feature'], d['channel']) for d in self.config["connexions of interest"]]


    def process(self):
        if self.is_processed: return self.new_anim

        vanim_ref, vanim_target = self.video_ref.to_animation(), self.video_target.to_animation()
        banim = b_utils.get_animation(self.blender_object_name)

        self.internal = InternalProcess.InternalProcess(vanim_ref, vanim_target, banim)

        warps = []
        channels = []
        for (feature, channel) in self.connexions_of_interest:
            warps.append(self.internal.make_warp(feature=feature))
            channels.append(channel)

        self.new_anim = self.internal.make_new_anim(channels=channels, warps=warps)
        return self.new_anim


    def to_blender(self):
        self.process()
        b_utils.set_animation(self.blender_object_name, self.new_anim)

