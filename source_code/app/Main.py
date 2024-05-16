
import os
from typing import List

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import app.AbstractIO as absIO
import app.InternalProcess as InternalProcess
import app.VideoIO as VideoIO
import app.SoftwareIO as SoftIO
import app.visualisation as vis

Color = VideoIO.Color

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
                "is impulsive":True,
            },
        ],
        "edit in place": False,
    }


class Main(absIO.AbstractIO):

    WARP_INTERPOLATION = "linear"

    def __init__(self, directory, verbose=0):
        super(Main, self).__init__(directory, verbose)
        self.finalize_init(default_config)

        self.video_ref = VideoIO.VideoIO(directory=directory, video_name=self.video_reference_filename, extension=self.video_extension, verbose=verbose-1)
        self.video_target = VideoIO.VideoIO(directory=directory, video_name=self.video_target_filename, extension=self.video_extension, verbose=verbose-1)
        self.blender_scene = SoftIO.SoftIO(directory=directory, verbose=verbose-1)
        
        self.blender_scene.check_file(self.directory+self.blender_scene_filename)
        self.internals = None
    
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

        self.warps:List[List[InternalProcess.Warp.LinearWarp1D]] = [[] for _ in self.internals]
        self.channels:List[List[str]] = [[] for _ in self.internals]
        self.features:List[List[str]] = [[] for _ in self.internals]
        for connexion in self.connexions_of_interest:
            obj_name, feature, channel, is_impulsive = connexion["object name"], connexion["video feature"], connexion["channel"], connexion["is impulsive"]
            index = self.blender_scene.object_names.index(obj_name) ## costly
            internal = self.internals[index]
            if is_impulsive: new_warp = internal.make_simplified_warp(feature=feature, uncertainty_threshold=2., interpolation=Main.WARP_INTERPOLATION, verbose=self.verbose-1)
            else : new_warp = internal.make_warp(feature=feature, interpolation=Main.WARP_INTERPOLATION, verbose=self.verbose-1)
            # because we prefer sparse warps for impulsive signals and dense ones for continuous signals
            self.warps[index].append(new_warp)
            self.channels[index].append(channel)
            self.features[index].append(feature)

        zipped = zip(self.internals, self.warps, self.channels)
        self.new_anims = [internal.make_new_anim(channels=channel, warps=warp) for internal, warp, channel in zipped]
        self.is_processed = True
        return self.new_anims


    def to_blender(self):
        self.process()
        self.blender_scene.set_animations(self.new_anims, in_place=self.edit_in_place)


    def draw_diagrams(self, animation_index=0, save=True, show=False):
        internal, warp, channel, feature = self.internals[animation_index], self.warps[animation_index][-1], self.channels[animation_index][-1], self.features[animation_index][-1]
        dtw = internal.dtw
        feature_curve1, feature_curve2 = internal.vanim1.find(feature), internal.vanim2.find(feature)
        original_curve = self.blender_scene.get_animations()[animation_index].find(channel)
        edited_curve =  SoftIO.b_utils.get_animation("Ball_edited").find(channel) # because self.new_anims[animation_index].find(channel) does not retrieve the fcurve

        Color.reset()
        
        ## Video feature : original VS retake
        fig0 = make_subplots(rows=2, shared_xaxes=True, subplot_titles=[f'Video feature curve: initial', f'Video feature curve: retook'], vertical_spacing=0.1)
        feature_curve1.display(handles=False, style="lines", fig=fig0, col=1, row=1)
        feature_curve2.display(handles=False, style="lines", fig=fig0, col=1, row=2)
        fig0.update_layout(xaxis2_title="Time (frames)", yaxis1_title="Magnitude (~pixels)", yaxis2_title="Magnitude (~pixels)", showlegend=False)
        title0 = f'Comparison of the initial and retook video feature curve "{feature}"'

        ## Connexion : video feature curve VS animation curve
        fig7 = make_subplots(rows=2, shared_xaxes=True, subplot_titles=[f'Video feature curve: {feature}', f'Animation curve: {original_curve.fullname}'], vertical_spacing=0.1)
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
        title6 = f'Global editing process using "{internal.feature}" feature on "{channel}" channel'

        figures:List[go.Figure] = [fig0, fig1, fig2, fig3, fig4, fig5, fig6, fig7]
        titles = [title0, title1, title2, title3, title4, title5, title6, title7]

        if not os.path.exists(f'{self.directory}/out/') : os.mkdir(f'{self.directory}/out/')
        for figure,title in zip(figures, titles):
            figure.update_layout(title=title)
            filetitle = title.replace('"','')
            if save: figure.write_html(f'{self.directory}/out/{filetitle}.html')
            if show: figure.show()

        return figures
