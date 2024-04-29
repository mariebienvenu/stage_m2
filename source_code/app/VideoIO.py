
import os, json

import numpy as np
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import app.Video as Video
import app.Animation as Animation
import app.OpticalFlow as oflow
import app.maths_utils as m_utils
import app.visualisation as vis

Curve = Animation.Curve


def default_config(height, width, frame_count, fps):
    return {
        "spatial crop":{
            "x1":0,
            "x2":width,
            "y1":0,
            "y2":height,
        },
        "time crop":{
            "start":0,
            "stop":int(frame_count-1),
        },
        "image processing method":"gray",
        "background proportion":0.0,
        "frame rate":fps, #basically unused
    }


class VideoIO: # TODO VideoIO -- make it herit from AbstractIO


    def __init__(self, directory, video_name, extension='.mp4', verbose=0):

        self.video = Video.Video(f'{directory}/{video_name}'+extension, verbose=verbose)
        self.name = video_name
        self.directory = directory

        self.config = None
        self.config_loaded = False

        self.is_processed = False

        try:
            self.load_config()
            self.complete_config()
        except OSError:
            if verbose>0: print("Did not find config file ; reverting to default config.")
            self.make_default_config()

        assert self.config_loaded, "Error when initializing VideoIO object."

    def get_default_config(self):
        return default_config(
            self.video.frame_height,
            self.video.frame_width,
            self.video.frame_count,
            self.video.fps,
        )
    
    def make_default_config(self):
        self.config = self.get_default_config()
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
        assert self.config_loaded, "Cannot save config when no config is loaded."
        with open(self.directory+self.name+"_config.json", "w") as outfile:
            json.dump(self.config, outfile)

    
    def complete_config(self):
        default = self.get_default_config()
        for key, value in default.items():
            if key not in self.config:
                self.config[key] = value


    def has_already_been_processed(self):
        if not os.path.exists(self.anim_directory+'params.json'): return False
        with open(self.anim_directory+'params.json', 'r') as openfile:
            processed_config = json.load(openfile)
        for (key, value) in processed_config.items():
            if key not in self.config or value != self.config[key]: return False
        for (key, value) in self.config.items():
            if key not in processed_config or value != processed_config[key]: return False
        return True    


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
    def frame_rate(self) -> float:
        return self.config['frame rate']
    
    @property
    def oflow_len(self):
        return self.video.frame_count - 1
    
    @property
    def anim_directory(self):
        return self.directory + self.name + '/'
    

    def get_frame_times(self): # TODO VideoIO.get_frame_times() -- move to Video ?
        return np.array(list(range(self.oflow_len)), dtype=np.float64)/self.frame_rate
    

    def process(self, force=False, verbose=0):
        if self.is_processed and not force:
            if verbose>0: print("Video already processed (VideoIO.process already called).")
            return
        
        elif self.has_already_been_processed() and not force:
            if verbose>0: print("Video already processed (found data with matching config on disk).")
            anim = Animation.Animation.load(self.anim_directory)
            self.magnitude_means = anim.find('Oflow magnitude - mean').get_values()
            self.magnitude_stds = anim.find('Oflow magnitude - std').get_values()
            self.angle_means = anim.find('Oflow angle - mean').get_values()
            self.angle_stds = anim.find('Oflow angle - std').get_values()
            self.velocity_x = anim.find('Velocity X').get_values()
            self.velocity_y = anim.find('Velocity Y').get_values()
            self.position_x = anim.find('Location X').get_values()
            self.position_y = anim.find('Location Y').get_values()
            self.is_processed = True
            return ## TODO - bug : this does not take into account the time cropping ! time_crop and curves are no longer aligned
        
        if verbose>0: print("Video currently processing (optical flow computation).")

        N = self.oflow_len
        self.magnitude_means = np.zeros((N))
        self.magnitude_stds = np.zeros((N))
        self.angle_means = np.zeros((N))
        self.angle_stds = np.zeros((N))

        for index in tqdm(range(self.oflow_len), desc='Oflow computation'):
            frame_before = self.video.get_frame(index, image_processing=self.image_processing_method, crop=self.spatial_crop)
            frame_after = self.video.get_frame(index+1, image_processing=self.image_processing_method, crop=self.spatial_crop)
            flow = oflow.OpticalFlow.compute_oflow(frame_before, frame_after, use_degrees=True)
        
            mask = flow.get_mask(background_proportion=self.background_proportion)
            self.magnitude_means[index] = flow.get_measure(oflow.Measure.MAGNITUDE_MEAN, mask)
            self.magnitude_stds[index] = flow.get_measure(oflow.Measure.MAGNITUDE_STD, mask)
            self.angle_means[index] = flow.get_measure(oflow.Measure.ANGLE_MEAN, mask)
            self.angle_stds[index] = flow.get_measure(oflow.Measure.ANGLE_STD, mask)

        velocity_x, velocity_y = oflow.polar_to_cartesian(self.magnitude_means, -self.angle_means, degrees=True) # reverse angles because up is - in image space
        self.velocity_x, self.velocity_y = np.ravel(velocity_x), np.ravel(velocity_y)
        self.position_x, self.position_y = m_utils.integrale3(self.velocity_x, step=1), m_utils.integrale3(self.velocity_y, step=1) # TODO delete this -> integrals accumulate error, bad feature

        self.is_processed = True


    def draw_diagrams(self, fig:go.Figure=None, save=True, show=False, time_in_seconds=False, verbose=0): # can be drawn either in frame scale or seconds scale
        fig = make_subplots(rows=2, cols=3, subplot_titles=(
            "Amplitude du flux au cours du temps",
            "Vitesses au cours du temps" ,
            "Portraits de phase",
            "Angle du flux au cours du temps",
            "Positions au cours du temps",
            "Trajectoire",
        )) if fig is None else fig

        self.process(verbose=verbose-1)

        frame_times = self.get_frame_times() if time_in_seconds else np.array(list(range(self.oflow_len)))

        vis.magnitude_angle(frame_times, self.magnitude_means, self.magnitude_stds, self.angle_means, self.angle_stds, fig=fig, rows=[1,2], cols=[1,1])
        vis.add_curve(y=self.velocity_y, x=self.position_y, name="y'=f(y) - Portrait de phase de Y", fig=fig, col=3, row=1)
        vis.add_curve(y=self.velocity_x, x=self.position_x, name="x'=f(x) - Portrait de phase de X", fig=fig, col=3, row=1)
        vis.add_curve(y=self.velocity_y, x=frame_times, name="y=f(t) - Velocity along X axis", fig=fig, col=2, row=1)
        vis.add_curve(y=self.velocity_x, x=frame_times, name="x=f(t) - Velocity along X axis", fig=fig, col=2, row=1)
        vis.add_curve(y=self.position_y, x=frame_times, name="y=f(t) - Translation along X axis", fig=fig, col=2, row=2)
        vis.add_curve(y=self.position_x, x=frame_times, name="x=f(t) - Translation along X axis", fig=fig, col=2, row=2)
        vis.add_curve(y=self.position_y, x=self.position_x, name="y=f(x) - Trajectoire", fig=fig, col=3, row=2)

        rows, cols = (1,1,2,2), (1,2,1,2)
        start = self.time_crop[0]/self.frame_rate if time_in_seconds else self.time_crop[0]
        stop = self.time_crop[1]/self.frame_rate if time_in_seconds else self.time_crop[1]
        for row, col in zip(rows, cols):
            fig.add_vline(x=start, row=row, col=col)
            fig.add_vline(x=stop, row=row, col=col)

        fig.update_layout(title=f'Optical flow  - {self.name}')
        if save: fig.write_html(self.directory+f"/{self.name}_diagram.html")
        if show: fig.show()

        fig2 = vis.magnitude_angle(frame_times, self.magnitude_means, self.magnitude_stds, self.angle_means, self.angle_stds)
        rows2, cols2 = (1,2), (1,1)
        for row, col in zip(rows2, cols2):
            fig2.add_vline(x=start, row=row, col=col)
            fig2.add_vline(x=stop, row=row, col=col)
        if save: fig2.write_html(self.directory+f"/{self.name}_oflow.html")
        if show: fig2.show()

        fig3 = vis.add_curve(y=self.position_y, x=self.position_x, name="y=f(x) - Trajectoire")
        if save: fig3.write_html(self.directory+f"/{self.name}_trajectory.html")
        if show: fig3.show()

        return [fig, fig2, fig3]


    def to_animation(self, save=True, verbose=0): # always in frame scale
        self.process(verbose=verbose-1)
        times = np.array(list(range(self.oflow_len))) if self.magnitude_means.size==self.oflow_len else np.array(list(range(self.time_crop[0], self.time_crop[1]+1)))
        anim = Animation.Animation([
            Curve.Curve(np.vstack((times, self.magnitude_means)).T, fullname='Oflow magnitude - mean'),
            Curve.Curve(np.vstack((times, self.magnitude_stds)).T, fullname='Oflow magnitude - std'),
            Curve.Curve(np.vstack((times, self.angle_means)).T, fullname='Oflow angle - mean'),
            Curve.Curve(np.vstack((times, self.angle_stds)).T, fullname='Oflow angle - std'),
            Curve.Curve(np.vstack((times, self.velocity_x)).T, fullname='Velocity X'),
            Curve.Curve(np.vstack((times, self.velocity_y)).T, fullname='Velocity Y'),
            Curve.Curve(np.vstack((times, self.position_x)).T, fullname='Location X'),
            Curve.Curve(np.vstack((times, self.position_y)).T, fullname='Location Y'),
        ])
        anim.crop(start=self.time_crop[0], stop=self.time_crop[1])
        if save: 
            anim.save(self.anim_directory)
            with open(self.anim_directory+"params.json", "w") as outfile:
                json.dump(self.config, outfile)
        return anim


    def get_spatial_crop_input_from_user(self, save=True): # TODO VideoIO.get_spatial_crop_input_from_user() -- maybe should be here instead of in video ?
        crop = self.video.get_spatial_crop_input_from_user(self.spatial_crop)
        self.config['spatial crop'] = crop
        if save: self.save_config()
        return crop
    

    def auto_time_crop(self, patience=2, save=True, verbose=0):
        self.process(verbose=verbose-1)
        times = np.array(list(range(self.oflow_len)))
        curve = Curve.Curve(np.vstack((times, self.magnitude_means)).T)
        start, stop = curve.get_auto_crop(use_handles=False, patience=patience)
        self.config['time crop'] = {'start':int(start), 'stop':int(stop)} # int32 -> int because int32 not json serialisable
        if save: self.save_config()
        return start, stop
    
    def record_video(self):
        frame_count = self.time_crop[1]-self.time_crop[0]+1
        frame_height = self.spatial_crop["y2"]-self.spatial_crop["y1"]
        frame_width = self.spatial_crop["x2"]-self.spatial_crop["x1"]
        content = np.zeros((frame_count, frame_height, frame_width))
        for i,index in enumerate(range(self.time_crop[0], self.time_crop[1]+1)):
            frame = self.video.get_frame(index, image_processing=self.image_processing_method, crop=self.spatial_crop)
            content[i,:,:] = np.copy(frame)
        video = Video.Video.from_array(content, self.directory+f'/_{self.name}_preprocessed.mp4', fps=self.video.fps)
        return video

        