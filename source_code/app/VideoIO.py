
import os, json

import numpy as np
from tqdm import tqdm
from plotly.subplots import make_subplots

import app.Video as Video
import app.Animation as Animation
import app.maths_utils as m_utils
import app.visualisation as vis

oflow = Video.oflow
Curve = Animation.Curve


def default_config(height, width, frame_count):
    return {
        "spatial crop":{
            "x1":0,
            "x2":width,
            "y1":0,
            "y2":height,
        },
        "time crop":{
            "start":0,
            "stop":frame_count-1,
        },
        "image processing method":"gray",
        "background proportion":0.97,
    }


class VideoIO:


    def __init__(self, directory, video_name, verbose=0):

        self.video = Video.Video(f'{directory}/{video_name}'+'.mp4', verbose=verbose)
        self.name = video_name
        self.directory = directory

        self.config = None
        self.config_loaded = False

        self.is_processed = False

        try:
            self.load_config()
        except OSError:
            print("Did not find config file ; reverting to default config.") if verbose>0 else None
            self.make_default_config()

        assert self.config_loaded, "Error when initializing VideoIO object."

    
    def make_default_config(self):
        self.config = default_config(
            self.video.frame_height,
            self.video.frame_width,
            self.video.frame_count,
        )
        self.config_loaded = True


    def load_config(self, force=False):
        if self.config_loaded and not force:
            return
        if not os.path.exists(self.directory+self.name+"_config.json"):
            raise OSError("No config file.")
        with open(self.directory+self.name+"_config.json", 'r') as openfile:
            self.config = json.load(openfile)
        self.config_loaded = True


    def save_config(self):
        assert self.config_loaded, "Cannot save config if no config is loaded."
        with open(self.directory+self.name+"_config.json", "w") as outfile:
            json.dump(self.config, outfile)


    @property # reminder : properties are read-only. The config file should be directly modified ; there is no reason to do it programmatically.
    def image_processing_method(self) -> str:
        return self.config['image processing method']
    
    @property
    def time_crop(self) -> tuple[float, float]:
        return self.config['time crop']['start'], self.config['time crop']['stop']
    
    @property
    def spatial_crop(self) -> dict:
        return self.config['spatial crop']
    
    @property
    def background_proportion(self) -> float:
        return self.config['background proportion']
    
    @property
    def oflow_len(self):
        return self.video.frame_count - 1
    

    def get_frame_times(self): # TODO move to Video ?
        return np.array(list(range(self.oflow_len)), dtype=np.float64)/self.video.fps

    
    def process(self, force=True):
        if self.is_processed and not force:
            return

        oflow_result = [self.video.get_optical_flow(
            index,
            image_processing=self.image_processing_method,
            crop=self.spatial_crop,
            background_proportion=self.background_proportion
        ) for index in tqdm(range(self.oflow_len), desc='Oflow computation')] # takes time

        oflow_measures = [res[2] for res in oflow_result]

        self.magnitude_means = np.array([measure["magnitude_mean"] for measure in oflow_measures])
        self.magnitude_stds = np.array([measure["magnitude_std"] for measure in oflow_measures])
        self.angle_means = np.array([measure["angle_mean"] for measure in oflow_measures])
        self.angle_stds = np.array([measure["angle_std"] for measure in oflow_measures])

        velocity_x, velocity_y = oflow.polar_to_cartesian(self.magnitude_means, -self.angle_means, degrees=True) # reverse angles because up is - in image space
        self.velocity_x, self.velocity_y = np.ravel(velocity_x), np.ravel(velocity_y)
        self.position_x, self.position_y = m_utils.integrale3(self.velocity_x, step=1), m_utils.integrale3(self.velocity_y, step=1)

        self.is_processed = True


    def draw_diagrams(self, fig=None, save=True, show=False):
        fig = make_subplots(rows=2, cols=3, subplot_titles=(
            "Amplitude du flux au cours du temps",
            "Vitesses au cours du temps" ,
            "Portraits de phase",
            "Angle du flux au cours du temps",
            "Positions au cours du temps",
            "Trajectoire",
        )) if fig is None else fig

        if not self.is_processed:
            self.process()

        frame_times = self.get_frame_times()

        vis.magnitude_angle(frame_times, self.magnitude_means, self.magnitude_stds, self.angle_means, self.angle_stds, fig=fig, rows=[1,2], cols=[1,1])
        vis.add_curve(y=self.velocity_y, x=self.position_y, name="y'=f(y) - Portrait de phase de Y", fig=fig, col=3, row=1)
        vis.add_curve(y=self.velocity_x, x=self.position_x, name="x'=f(x) - Portrait de phase de X", fig=fig, col=3, row=1)
        vis.add_curve(y=self.velocity_y, x=frame_times, name="y=f(t) - Velocity along X axis", fig=fig, col=2, row=1)
        vis.add_curve(y=self.velocity_x, x=frame_times, name="x=f(t) - Velocity along X axis", fig=fig, col=2, row=1)
        vis.add_curve(y=self.position_y, x=frame_times, name="y=f(t) - Translation along X axis", fig=fig, col=2, row=2)
        vis.add_curve(y=self.position_x, x=frame_times, name="x=f(t) - Translation along X axis", fig=fig, col=2, row=2)
        vis.add_curve(y=self.position_y, x=self.position_x, name="y=f(x) - Trajectoire", fig=fig, col=3, row=2)

        rows, cols = (1,1,2,2), (1,2,1,2)
        for row, col in zip(rows, cols):
            fig.add_vline(x=self.time_crop[0], row=row, col=col)
            fig.add_vline(x=self.time_crop[1], row=row, col=col)

        fig.update_layout(title=f'Optical flow  - {self.name}')
        fig.write_html(self.directory+f"/{self.name}_diagram.html") if save else None
        fig.show() if show else None

        return fig


    def to_animation(self, save=True):
        if not self.is_processed:
            self.process()
        frame_times = self.get_frame_times()
        anim = Animation.Animation([
            Curve.Curve(np.vstack((frame_times, self.magnitude_means)).T, fullname='Oflow magnitude - mean'),
            Curve.Curve(np.vstack((frame_times, self.magnitude_stds)).T, fullname='Oflow magnitude - std'),
            Curve.Curve(np.vstack((frame_times, self.angle_means)).T, fullname='Oflow angle - mean'),
            Curve.Curve(np.vstack((frame_times, self.angle_stds)).T, fullname='Oflow angle - std'),
            Curve.Curve(np.vstack((frame_times, self.velocity_x)).T, fullname='Velocity X'),
            Curve.Curve(np.vstack((frame_times, self.velocity_y)).T, fullname='Velocity Y'),
            Curve.Curve(np.vstack((frame_times, self.position_x)).T, fullname='Location X'),
            Curve.Curve(np.vstack((frame_times, self.position_y)).T, fullname='Location Y'),
        ])
        anim.crop(start=self.time_crop[0], stop=self.time_crop[1])
        anim.save(self.directory + self.name + '/') if save else None
        return anim


    def get_spatial_crop_input_from_user(self, save=True):
        crop = self.video.get_spatial_crop_input_from_user()
        self.config['spatial crop'] = crop
        self.save_config() if save else None
        return crop
    

    def auto_time_crop(self, patience=2, save=True):
        if not self.is_processed:
            self.process()
        start, stop = oflow.get_crop(self.get_frame_times(), self.magnitude_means)
        self.config['time crop'] = {'start':start, 'stop':stop}
        self.save_config() if save else None
        return start, stop