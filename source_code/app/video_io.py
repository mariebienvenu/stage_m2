
import os, json

import numpy as np
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from app.video import Video
from app.animation import Animation
from app.optical_flow import OpticalFlow
import app.optical_flow as oflow
import app.maths_utils as m_utils
import app.visualisation as vis
from app.abstract_io import AbstractIO

from app.curve import Curve
from app.color import Color


def default_config(height, width, frame_count, fps):
    return {
        "spatial crops":[{
            "x1":0,
            "x2":width,
            "y1":0,
            "y2":height,
        }],
        "time crop":{
            "start":0,
            "stop":int(frame_count-1),
        },
        "image processing method":"gray",
        "background proportion":0.0,
        "frame rate":fps, #basically unused
    }


class VideoIO(AbstractIO):


    def __init__(self, directory:str, video_name:str, extension='.mp4', verbose=0):
        super(VideoIO, self).__init__(directory, verbose)
        self.video = Video(f'{directory}/{video_name}'+extension, verbose=verbose)
        self.name = video_name
        def maker():
            return default_config(
                self.video.frame_height,
                self.video.frame_width,
                self.video.frame_count,
                self.video.fps,
            )
        self.finalize_init(maker)

    @property
    def config_filename(self):
        return self.directory+self.name+"_config.json"


    def has_already_been_processed(self): # TODO: check if this handles well spatial_crops now that it is a list of dictionnaries
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
    def spatial_crops(self) -> list[dict]:
        return self.config['spatial crops']
    
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
    
    @property
    def oflow_frame_times(self):
        return self.video.frame_times[:-1]
    

    def get_oflow(self, frame_index:int, crop_index=0): #  TODO test in dedicated file + put it into self.process
        crop = self.spatial_crops[crop_index]
        frame_before = self.video.get_frame(frame_index, image_processing=self.image_processing_method, crop=crop)
        frame_after = self.video.get_frame(frame_index+1, image_processing=self.image_processing_method, crop=crop)
        flow = OpticalFlow.compute_oflow(frame_before, frame_after, use_degrees=True)
        return flow        


    def process(self, force=False):
        if self.is_processed and not force:
            if self.verbose>0: print("Video already processed (VideoIO.process already called).")
            return
        
        elif self.has_already_been_processed() and not force:
            if self.verbose>0: print("Video already processed (found data with matching config on disk).")
            anim = Animation.load(self.anim_directory)
            self.data = {}
            for curve in anim:
                self.data[curve.fullname] = curve.get_values()
            self.is_processed = True
            return ## TODO - bug : this does not take into account the time cropping ! time_crop and curves are no longer aligned
        
        if self.verbose>0: print("Video currently processing (optical flow computation).")

        N = self.oflow_len
        self.data = {}
        for i in range(len(self.spatial_crops)):
            for key in ['Oflow magnitude - mean', 'Oflow magnitude - std', 'Oflow angle - mean', 'Oflow angle - std']:
                self.data[f'{key} {i+1}'] = np.zeros((N))

        for i, crop in enumerate(self.spatial_crops):
            for index in tqdm(range(self.oflow_len), desc='Oflow computation'):
                frame_before = self.video.get_frame(index, image_processing=self.image_processing_method, crop=crop)
                frame_after = self.video.get_frame(index+1, image_processing=self.image_processing_method, crop=crop)
                flow = OpticalFlow.compute_oflow(frame_before, frame_after, use_degrees=True)
            
                mask = flow.get_mask(background_proportion=self.background_proportion)
                self.data[f'Oflow magnitude - mean {i+1}'][index] = flow.get_measure(oflow.Measure.MAGNITUDE_MEAN, mask)
                self.data[f'Oflow magnitude - std {i+1}'][index] = flow.get_measure(oflow.Measure.MAGNITUDE_STD, mask)
                self.data[f'Oflow angle - mean {i+1}'][index] = flow.get_measure(oflow.Measure.ANGLE_MEAN, mask)
                self.data[f'Oflow angle - std {i+1}'][index] = flow.get_measure(oflow.Measure.ANGLE_STD, mask)

            velocity_x, velocity_y = oflow.polar_to_cartesian(self.data[f'Oflow magnitude - mean {i+1}'], -self.data[f'Oflow angle - mean {i+1}'], degrees=True) # reverse angles because up is - in image space
            self.data[f"Velocity X {i+1}"], self.data[f"Velocity Y {i+1}"] = np.ravel(velocity_x), np.ravel(velocity_y)
            self.data[f"Location X {i+1}"], self.data[f"Location Y {i+1}"] = m_utils.integrale3(self.data[f"Velocity X {i+1}"], step=1), m_utils.integrale3(self.data[f"Velocity Y {i+1}"], step=1) 
            # TODO delete this -> integrals accumulate error, bad feature -> eh ? maybe not always, second derivative of this is better than first derivative of velocity...

        self.is_processed = True

    
    def draw_diagrams(self, fig:go.Figure=None, save=True, show=False, time_in_seconds=False):
        res:list[go.Figure] = []
        for i,_ in enumerate(self.spatial_crops):
            res += self.draw_diagrams_i(fig, save, show, time_in_seconds, i)
        return res
    

    def draw_diagrams_i(self, fig:go.Figure=None, save=True, show=False, time_in_seconds=False, i=0): # can be drawn either in frame scale or seconds scale
        fig = make_subplots(rows=2, cols=3, subplot_titles=(
            "Amplitude du flux au cours du temps",
            "Vitesses au cours du temps" ,
            "Portraits de phase",
            "Angle du flux au cours du temps",
            "Positions au cours du temps",
            "Trajectoire",
        )) if fig is None else fig

        self.process()
        data = {key[:-2]:value for key,value in self.data.items() if str(i+1) in key}

        frame_times = self.oflow_frame_times if time_in_seconds else np.array(list(range(self.oflow_len)))
        Color.reset()

        vis.magnitude_angle(frame_times, data['Oflow magnitude - mean'], data['Oflow magnitude - std'], data['Oflow angle - mean'], data['Oflow angle - std'], fig=fig, rows=[1,2], cols=[1,1])
        vis.add_curve(y=data["Velocity Y"], x=data["Location Y"], name="y'=f(y) - Portrait de phase de Y", fig=fig, col=3, row=1)
        vis.add_curve(y=data["Velocity X"], x=data["Location X"], name="x'=f(x) - Portrait de phase de X", fig=fig, col=3, row=1)
        vis.add_curve(y=data["Velocity Y"], x=frame_times, name="y'=f(t) - Velocity along Y axis", fig=fig, col=2, row=1)
        vis.add_curve(y=data["Velocity X"], x=frame_times, name="x'=f(t) - Velocity along X axis", fig=fig, col=2, row=1)
        vis.add_curve(y=data["Location Y"], x=frame_times, name="y=f(t) - Translation along Y axis", fig=fig, col=2, row=2)
        vis.add_curve(y=data["Location X"], x=frame_times, name="x=f(t) - Translation along X axis", fig=fig, col=2, row=2)
        vis.add_curve(y=data["Location Y"], x=data["Location X"], name="y=f(x) - Trajectoire", fig=fig, col=3, row=2)

        rows, cols = (1,1,2,2), (1,2,1,2)
        start = self.time_crop[0]/self.frame_rate if time_in_seconds else self.time_crop[0]
        stop = self.time_crop[1]/self.frame_rate if time_in_seconds else self.time_crop[1]
        for row, col in zip(rows, cols):
            fig.add_vline(x=start, row=row, col=col)
            fig.add_vline(x=stop, row=row, col=col)

        fig.update_layout(title=f'Optical flow  - {self.name}')
        if save: fig.write_html(self.directory+f"/{self.name}_diagram_{i+1}.html")
        if show: fig.show()

        fig2 = vis.magnitude_angle(frame_times, data['Oflow magnitude - mean'], data['Oflow magnitude - std'], data['Oflow angle - mean'], data['Oflow angle - std'])
        rows2, cols2 = (1,2), (1,1)
        for row, col in zip(rows2, cols2):
            fig2.add_vline(x=start, row=row, col=col)
            fig2.add_vline(x=stop, row=row, col=col)
        fig2.update_layout(
            title=f'Optical flow  - {self.name}',
            yaxis1_title="magnitude (pixels/frame)",
            yaxis2_title="angle (degrees/frame)",
            xaxis2_title="time (frames)",
            showlegend=False,
        )
        if save: fig2.write_html(self.directory+f"/{self.name}_oflow_{i+1}.html")
        if show: fig2.show()

        fig3 = vis.add_curve(y=data["Location Y"], x=data["Location X"], name="y=f(x) - Trajectoire")
        if save: fig3.write_html(self.directory+f"/{self.name}_trajectory_{i+1}.html")
        if show: fig3.show()

        fig4 = vis.add_curve(y=m_utils.derivee_seconde(data["Location Y"], step=1), x=frame_times, name='y"=f(t) - Acceleration along Y axis')
        vis.add_curve(y=m_utils.derivee_seconde(data["Location X"], step=1), x=frame_times, name='x"=f(t) - Acceleration along X axis', fig=fig4)
        fig4.update_layout(title=f'Acceleration  - {self.name}', xaxis_title="time (frames)", yaxis_title="magnitude (pixels/frame²)")
        if save: fig4.write_html(self.directory+f"/{self.name}_acceleration_{i+1}.html")
        if show: fig4.show()

        return [fig, fig2, fig3, fig4]


    def to_animation(self, save=True): # always in frame scale
        self.process()
        times = np.array(list(range(self.oflow_len))) if self.data['Oflow magnitude - mean 1'].size==self.oflow_len else np.array(list(range(self.time_crop[0], self.time_crop[1]+1)))
        anim = Animation([Curve(np.vstack((times, value)).T, fullname=key) for key,value in self.data.items()])
        anim.crop(start=self.time_crop[0], stop=self.time_crop[1])
        if save: 
            anim.save(self.anim_directory)
            with open(self.anim_directory+"params.json", "w") as outfile:
                json.dump(self.config, outfile)
        return anim


    def get_spatial_crop_input_from_user(self, save=True): # TODO VideoIO.get_spatial_crop_input_from_user() -- maybe should be here instead of in video ?
        crops = self.video.get_spatial_crop_input_from_user(self.spatial_crops)
        self.config['spatial crops'] = crops
        if save: self.save_config()
        return crops
    

    def auto_time_crop(self, patience=2, save=True):
        self.process()
        times = np.array(list(range(self.oflow_len)))
        curve = Curve(np.vstack((times, self.data['Oflow magnitude - mean 1'])).T)
        start, stop = curve.get_auto_crop(use_handles=False, patience=patience)
        self.config['time crop'] = {'start':int(start), 'stop':int(stop)} # int32 -> int because int32 not json serialisable
        if save: self.save_config()
        return start, stop
    

    def record_video(self):
        videos = []
        for j,crop in enumerate(self.spatial_crops):
            frame_count = self.time_crop[1]-self.time_crop[0]+1
            frame_height = crop["y2"]-crop["y1"]
            frame_width = crop["x2"]-crop["x1"]
            content = np.zeros((frame_count, frame_height, frame_width))
            for i,index in enumerate(range(self.time_crop[0], self.time_crop[1]+1)):
                frame = self.video.get_frame(index, image_processing=self.image_processing_method, crop=crop)
                content[i,:,:] = np.copy(frame)
            video = Video.from_array(content, self.directory+f'/_{self.name}_preprocessed_{j+1}.mp4', fps=self.video.fps)
            videos.append(video)
        return videos

        