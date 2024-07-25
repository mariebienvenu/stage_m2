
import os

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from app.abstract_io import AbstractIO
from app.internal_process import InternalProcess
from app.video_io import VideoIO
from app.dcc_io import SoftIO
import app.visualisation as vis
from app.color import Color
import app.warping as warping
import app.blender_utils as b_utils
from app.animation import Animation
from app.curve import Curve

def default_config():
    return {
        "video extension": ".mp4",
        "video reference filename": "ref",
        "video target filename": "target",
        "blender scene filename": "scene.blend",
        "connexions of interest": [
            {
                "object name": "Ball",
                "channel": "Location Y",
                "video feature": "First derivative of Velocity Y",
                "is impulsive": True,
            },
        ],
        "edit in place": False,
        "temporal offset": 0, # offset to add to video curves to be aligned in time with the animation curves ; should be small (less than 5 frames).
    }


class Main(AbstractIO):

    WARP_INTERPOLATION = "linear" # the way the warping is interpolated between matches
    DTW_CONSTRAINTS_LOCAL = 10 # if 0 : dtw constraint computation is global ; else, it is local with a range of DTW_CONSTRAINTS_LOCAL (in frames)
    SPOT_FOR_DTW_CONSTRAINTS = False
    USE_SEMANTIC = True

    def __init__(self, directory, no_blender=False, verbose=0):
        super(Main, self).__init__(directory, verbose)
        self.finalize_init(default_config)
        self.load_videos()
        self.no_blender = no_blender
        if not no_blender:
            self.blender_scene = SoftIO(directory=directory, verbose=verbose-1)
            self.blender_scene.check_file(self.directory+self.blender_scene_filename)
    
    @property
    def config_filename(self) -> str:
        return self.directory + "main_config.json"
    
    @property
    def connexions_of_interest(self) -> list[dict]:
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

    @property
    def temporal_offset(self) -> int:
        return self.config["temporal offset"]
    

    def load_videos(self):
        self.video_ref = VideoIO(directory=self.directory, video_name=self.video_reference_filename, extension=self.video_extension, verbose=self.verbose-1)
        self.video_target = VideoIO(directory=self.directory, video_name=self.video_target_filename, extension=self.video_extension, verbose=self.verbose-1)
    

    def process(self, force=False):
        if self.is_processed and not force: return self.new_anims

        vanim_ref, vanim_target = self.video_ref.to_animation(), self.video_target.to_animation()
        if not self.no_blender: vanim_ref.time_transl(self.blender_scene.start-vanim_ref.time_range[0]-1+self.temporal_offset) # -1 parce que en dérivant on décale de une frame donc si on veut rester sur start...
        if not self.no_blender: vanim_target.time_transl(self.blender_scene.start-vanim_target.time_range[0]-1+self.temporal_offset)
        vanim_ref.update_time_range()
        vanim_target.update_time_range()
        vanim_ref.enrich()
        vanim_target.enrich()
        if not self.no_blender: banims = self.blender_scene.get_animations()
        else : banims = [Animation()]
        for i,banim in enumerate(banims):
            banim.save(self.directory+f'/{i}/')
        #for banim in banims:
        #    banim.enrich() # TODO -- will be useful when we automatically decide of connexions based on multi-modal correlations

        self.internals = [
            InternalProcess(vanim_ref, vanim_target, banim, verbose=self.verbose-1) for banim in banims
        ]
        self.figures_and_titles:list[list[tuple[list[go.Figure],list[str]]]] = []

        self.is_impulsive:list[list[bool]] = [[] for _ in self.internals]
        self.channels:list[list[list[str]]] = [[] for _ in self.internals]
        self.features:list[list[str]] = [[] for _ in self.internals]
        for connexion in self.connexions_of_interest:
            obj_name, feature, channel, is_impulsive = connexion["object name"], connexion["video feature"], connexion["channel"], connexion["is impulsive"]
            internal_index = self.blender_scene.object_names.index(obj_name) if not self.no_blender else 0 ## costly
            self.is_impulsive[internal_index].append(is_impulsive)  # we prefer sparse warps for impulsive signals and dense ones for continuous signals
            if feature not in self.features[internal_index]:
                self.features[internal_index].append(feature)
                self.channels[internal_index].append([])
            feature_index = self.features[internal_index].index(feature)
            self.channels[internal_index][feature_index].append(channel)

        zipped = zip(self.internals, self.features, self.is_impulsive, self.channels)
        self.new_anims = []
        for obj_index, (internal, features, is_impulsive, channels_list) in enumerate(zipped):
            self.figures_and_titles.append([])
            new_anim = None
            for feature_index, (feature, channels) in enumerate(zip(features, channels_list)):
                    new_anim = internal.process(feature=feature, channels=channels,filter_indexes=is_impulsive, warp_interpolation=Main.WARP_INTERPOLATION, spot_for_dtw_constraint=Main.SPOT_FOR_DTW_CONSTRAINTS, use_semantic=Main.USE_SEMANTIC)
                    self.figures_and_titles[obj_index].append(self.draw_diagrams(obj_index, feature_index))
            self.new_anims.append(new_anim)
        #self.new_anims = [internal.process(feature=feature, channels=channels,filter_indexes=is_impulsive, warp_interpolation=Main.WARP_INTERPOLATION, spot_for_dtw_constraint=Main.SPOT_FOR_DTW_CONSTRAINTS) for internal, feature, is_impulsive, channels in zipped]
        self.is_processed = True
        return self.new_anims


    def to_blender(self):
        assert not self.no_blender, "Impossible to send to blender when no_blender is set to True."
        self.process()
        self.blender_scene.set_animations(self.new_anims, in_place=self.edit_in_place)


    def draw_diagrams(self, object_index=0, feature_index=0):

        internal, channels, feature = self.internals[object_index], self.channels[object_index][feature_index], self.features[object_index][feature_index]
        channel = channels[-1]
        warp = internal.warp
        dtw = internal.dtw
        feature_curve1, feature_curve2 = internal.vanim1.find(feature), internal.vanim2.find(feature)
        original_curve = self.blender_scene.get_animations()[object_index].find(channel) if not self.no_blender else Curve()
        obj_name = self.connexions_of_interest[0]["object name"]
        edited_curve =  b_utils.get_animation(f"{obj_name}{self.blender_scene.edited_suffix}").find(channel)  if not self.no_blender else Curve() # because self.new_anims[animation_index].find(channel) does not retrieve the fcurve
        
        original_curve = Animation([self.blender_scene.get_animations()[object_index].find(channel) for channel in channels]) if not self.no_blender else Animation()
        edited_curve =  Animation([b_utils.get_animation(f"{obj_name}{self.blender_scene.edited_suffix}").find(channel) for channel in channels])  if not self.no_blender else Animation()

        Color.reset()
        
        ## Video feature : original VS retake
        fig0 = make_subplots(rows=2, shared_xaxes=True, subplot_titles=[f'Video feature curve: initial', f'Video feature curve: retook'], vertical_spacing=0.1)
        feature_curve1.display(handles=False, style="lines", fig=fig0, col=1, row=1)
        feature_curve2.display(handles=False, style="lines", fig=fig0, col=1, row=2)
        fig0.update_layout(xaxis2_title="Time (frames)", yaxis1_title="Magnitude (~pixels)", yaxis2_title="Magnitude (~pixels)", showlegend=False)
        title0 = f'Comparison of the initial and retook video feature curve "{feature}"'

        ## Connexion : video feature curve VS animation curve
        fig7 = make_subplots(rows=2, shared_xaxes=True, subplot_titles=[f'Video feature curve: {feature}', f'Animation curve: {obj_name}'], vertical_spacing=0.1)
        feature_curve1.display(handles=False, style="lines", fig=fig7, col=1, row=1)
        original_curve.display(fig=fig7, col=1, row=2)
        original_curve.sample().display(fig=fig7, row=2, col=1, handles=False, style='lines')
        fig7.update_layout(xaxis2_title="Time (frames)", yaxis1_title="Magnitude (~pixels)", yaxis2_title="Magnitude (Blender unit)", showlegend=False)
        title7 = "Comparison of the original animation curve and initial video feature curve"

        ## DTW cost matrix
        fig1 = vis.add_heatmap(pd.DataFrame(dtw.cost_matrix))
        fig1.update_layout(xaxis_title="Time (frames) - retook", yaxis_title="Time (frames) - initial")
        title1 = "DTW cost matrix"

        ## Bijections
        fig2 = vis.add_curve(y=dtw.bijection[1], x=dtw.bijection[0], name="DTW matches", style="lines")
        vis.add_curve(y=warp.output_data, x=warp.input_data, style="markers", fig=fig2, name="Warp reference points")
        x = np.arange(warp.input_data[0], warp.input_data[-1]+1, 1)
        vis.add_curve(y=warp(x,None)[0], x=x, style="lines", fig=fig2, name="Warp function")
        fig2.update_layout(xaxis_title="Time (frames) - initial", yaxis_title="Time (frames) - retook")
        title2 = "Time bijection"

        ## DTW pairs
        fig3 = vis.add_curve(y=dtw.values2+3, x=dtw.times2, name='retook')
        vis.add_curve(y=dtw.values1, x=dtw.times1, name='initial', fig=fig3)
        vis.add_pairings(y1=dtw.values1, y2=dtw.values2+3, pairs=dtw.pairings, x1=dtw.times1, x2=dtw.times2, fig=fig3)
        fig3.update_layout(xaxis_title="Time (frames)", yaxis_title="Magnitude (~pixels)")
        title3 = f'Matches between initial and retook feature curve "{feature}"'

        ## DTW local constraints along the path
        fig4 = vis.add_curve(y=internal.dtw_constraints, name="local - chosen")
        vis.add_curve(y=dtw.global_constraints(), name="global", fig=fig4)
        fig4.update_layout(xaxis_title="DTW path (pairs over time)", yaxis_title="Avoided cost (~pixels)")
        title4 = "Constraint on DTW chosen path over time"

        ## Animation curves : Edited curve VS original one
        fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Original curves", "Edited curves"], vertical_spacing=0.1)
        original_curve.display(fig=fig5, row=1, col=1)
        edited_curve.display(fig=fig5, row=2, col=1)
        original_curve.sample().display(fig=fig5, row=1, col=1, handles=False, style='lines')
        edited_curve.sample().display(fig=fig5, row=2, col=1, handles=False, style='lines')
        fig5.update_layout(xaxis2_title="Time (frames)", yaxis1_title="Magnitude (Blender units)", yaxis2_title="Magnitude (Blender units)", showlegend=False)
        title5 = "Comparison of the original and post-editing animation curves"

        ## Global visualisation
        fig6 = make_subplots(rows=2, cols=2, shared_xaxes="all", row_titles=["Video curves", "Animation curves"], column_titles=["Before", "After"], vertical_spacing=0.1, horizontal_spacing=0.1)
        feature_curve1.display(handles=False, style="lines", row=1, col=1, fig=fig6)
        feature_curve2.display(handles=False, style="lines", row=1, col=2, fig=fig6)
        original_curve.display(fig=fig6, row=2, col=1)
        edited_curve.display(fig=fig6, row=2, col=2)
        original_curve.sample().display(fig=fig6,  row=2, col=1, handles=False, style='lines')
        edited_curve.sample().display(fig=fig6, row=2, col=2, handles=False, style='lines')
        fig6.update_layout(xaxis3_title="Time (frames)", xaxis4_title="Time (frames)", yaxis1_title="Magnitude (~pixels)", yaxis3_title="Magnitude (Blender units)", showlegend=False)
        title6 = f'Global editing process using "{feature}" feature on "{channel}" channel' if len(channels)==1 else f'Global editing process using "{feature}" feature on {len(channels)} channels of {obj_name}'

        figures:list[go.Figure] = [fig0, fig1, fig2, fig3, fig4, fig5, fig6, fig7]
        titles = [title0, title1, title2, title3, title4, title5, title6, title7]

        other_figures, other_titles = internal.make_diagrams(number_issues=False, anim_style="lines+markers") # TODO should not be like this

        return figures+other_figures, titles+other_titles


    def display(self, save=True, show=False, directory=None):
        if directory is None: directory = self.directory + '/out/'
        if not os.path.exists(directory) : os.mkdir(directory)
        object_index, feature_index = 0, 0
        for liste in self.figures_and_titles:
            for figures, titles in liste:
                for figure,title in zip(figures, titles):
                    figure.update_layout(title=title)
                    filetitle = title.replace('"','')
                    if save: figure.write_html(f'{directory}/{filetitle}_{object_index}_{feature_index}.html')
                    if show: figure.show()
                feature_index += 1
            object_index += 1
            feature_index = 0



def for_the_paper(main:Main, object_index=0, feature_index=0, save=True, show=False):
    ## What we need
    # S_in
    # S_out
    # Heatmap with path [and circled matches ?]
    # Matches between S_in and S_out
    # Constraints along best path
    internal, channels, feature = main.internals[object_index], main.channels[object_index][feature_index], main.features[object_index][feature_index]
    channel = channels[-1]
    warp = internal.warp
    dtw = internal.dtw
    feature_curve1, feature_curve2 = internal.vanim1.find(feature), internal.vanim2.find(feature)
    Color.reset()
    '''
    ## Video feature : original VS retake
    fig0 = make_subplots(rows=2, shared_xaxes=True, subplot_titles=[f'Video feature curve: initial', f'Video feature curve: retook'], vertical_spacing=0.1)
    feature_curve1.display(handles=False, style="lines", fig=fig0, col=1, row=1)
    feature_curve2.display(handles=False, style="lines", fig=fig0, col=1, row=2)
    fig0.update_layout(xaxis2_title="Time (frames)", yaxis1_title="Magnitude (~pixels)", yaxis2_title="Magnitude (~pixels)", showlegend=False)
    title0 = f'Comparison of the initial and retook video feature curve "{feature}"'

    ## Video feature : original VS retake
    fig0 = make_subplots(rows=2, shared_xaxes=True, subplot_titles=[f'Video feature curve: initial', f'Video feature curve: retook'], vertical_spacing=0.1)
    feature_curve1.display(handles=False, style="lines", fig=fig0, col=1, row=1)
    feature_curve2.display(handles=False, style="lines", fig=fig0, col=1, row=2)
    fig0.update_layout(xaxis2_title="Time (frames)", yaxis1_title="Magnitude (~pixels)", yaxis2_title="Magnitude (~pixels)", showlegend=False)
    title0 = f'Comparison of the initial and retook video feature curve "{feature}"'
    '''
    inlier_color, outlier_color, path_color = Color.next(), Color.next(), Color.next()
    ref_color, target_color = Color.next(), Color.next()
    constraints = internal.dtw_constraints
    distances = dtw.alternate_path_differences()
    pairs = dtw.pairings
    inliers = internal.kept_indexes
    outliers = getattr(internal, "outliers", [])
    vertical_offset = 10

    x1, y1, x2, y2 = dtw.times1, dtw.values1, dtw.times2, dtw.values2+vertical_offset
    all_pairings = pairs
    refined_pairings = [e for i,e in enumerate(pairs) if i in inliers]

    ## Matches selected
    fig = vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=all_pairings, color=(210, 210, 210), opacity=1)
    vis.add_curve(y=y2, x=x2, name="curve1", color=target_color, fig=fig)
    vis.add_curve(y=y1, x=x1, name="curve2", color=ref_color, fig=fig)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=refined_pairings, color="green", opacity=1, fig=fig)

    fig.update_layout(
        xaxis_title="Time (frames)",
        yaxis_title="Amplitude (arbitrary)",
    )
    title = "Filtered correspondance"


    directory = main.directory + '/out/'
    if not os.path.exists(directory) : os.mkdir(directory)
    fig.update_layout(title=title)
    filetitle = title.replace('"','')
    if save: fig.write_html(f'{directory}/{filetitle}_{object_index}_{feature_index}.html')
    if show: fig.show()