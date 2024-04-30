
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

        self.warps:List[List[InternalProcess.Warp.LinearWarp1D]] = [[] for _ in self.internals]
        self.channels:List[List[str]] = [[] for _ in self.internals]
        for connexion in self.connexions_of_interest:
            obj_name, feature, channel = connexion["object name"], connexion["video feature"], connexion["channel"]
            index = self.blender_scene.object_names.index(obj_name) ## costly
            internal = self.internals[index]
            self.warps[index].append(internal.make_simplified_warp(feature=feature, verbose=self.verbose-1)) ## TODO add config to chose between dense and sparse warp based on impulsivity of the signal
            self.channels[index].append(channel)

        zipped = zip(self.internals, self.warps, self.channels)
        self.new_anims = [internal.make_new_anim(channels=channel, warps=warp) for internal, warp, channel in zipped]
        self.is_processed = True
        return self.new_anims


    def to_blender(self):
        self.process()
        self.blender_scene.set_animations(self.new_anims, in_place=self.edit_in_place)


    def draw_diagrams(self, animation_index=0, save=True, show=False):
        internal, warp, channel = self.internals[animation_index], self.warps[animation_index][-1], self.channels[animation_index][-1]
        dtw = internal.dtw

        ## DTW cost matrix
        fig1 = vis.add_heatmap(pd.DataFrame(dtw.cost_matrix))
        title1 = "DTW cost matrix"

        ## Bijections
        fig2 = vis.add_curve(y=dtw.bijection[1], x=dtw.bijection[0], name="DTW matches", style="lines")
        vis.add_curve(y=warp.Y, x=warp.X, style="markers", fig=fig2, name="Warp reference points")
        x = np.arange(warp.X[0], warp.X[-1]+1, 1)
        vis.add_curve(y=warp(x,None)[0], x=x, style="lines", fig=fig2, name="Warp function")
        title2 = "Time bijection"

        ## DTW pairs
        fig3 = vis.add_curve(y=dtw.values2+3, x=dtw.times2, name=dtw.curve2.fullname)
        vis.add_curve(y=dtw.values1, x=dtw.times1, name=dtw.curve1.fullname, fig=fig3)
        vis.add_pairings(y1=dtw.values1, y2=dtw.values2+3, pairs=dtw.pairings, x1=dtw.times1, x2=dtw.times2, fig=fig3)
        title3 = "Matches between reference and target feature curve"

        ## DTW local constraints
        fig4 = vis.add_curve(y=internal.dtw_constraints, x=dtw.bijection[0])
        title4 = "Constraint on DTW chosen path over time"

        ## Edited curve VS original one
        fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_titles=["Original curves", "Edited curves"])
        original_curve = self.blender_scene.get_animations()[animation_index].find(channel)
        edited_curve =  SoftIO.b_utils.get_animation("Ball_edited").find(channel) # because self.new_anims[animation_index].find(channel) does not retrieve the fcurve
        original_curve.display(fig=fig5, row=1, col=1)
        edited_curve.display(fig=fig5, row=2, col=1)
        original_curve.sample().display(fig=fig5, row=1, col=1, handles=False, style='lines')
        edited_curve.sample().display(fig=fig5, row=2, col=1, handles=False, style='lines')
        title5 = "Comparison of the original and post-editing animation curves"

        ## Global visualisation
        fig6 = make_subplots(rows=2, cols=2, shared_xaxes="all", row_titles=["Video curves", "Animation curves"], column_titles=["Before", "After"])
        vis.add_curve(y=dtw.values1, x=dtw.times1, name=dtw.curve1.fullname, row=1, col=1, fig=fig6)
        vis.add_curve(y=dtw.values2, x=dtw.times2, name=dtw.curve2.fullname, row=1, col=2, fig=fig6)
        original_curve.display(fig=fig6, row=2, col=1)
        edited_curve.display(fig=fig6, row=2, col=2)
        original_curve.sample().display(fig=fig6,  row=2, col=1, handles=False, style='lines')
        edited_curve.sample().display(fig=fig6, row=2, col=2, handles=False, style='lines')
        title6 = f"Global editing process using {internal.feature} feature on {channel} channel"

        figures:List[go.Figure] = [fig1, fig2, fig3, fig4, fig5, fig6]
        titles = [title1, title2, title3, title4, title5, title6]

        if not os.path.exists(f'{self.directory}/out/') : os.mkdir(f'{self.directory}/out/')
        for figure,title in zip(figures, titles):
            figure.update_layout(title=title)
            if save: figure.write_html(f'{self.directory}/out/{title}.html')
            if show: figure.show() # How to make the titles show ?

        return figures
