

import os
import json
import re

import plotly.express as px
import plotly

import app.visualisation as vis

def read_from_html(filename):
    with open(filename, encoding='utf-8') as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}    
    return plotly.io.from_json(json.dumps(plotly_json))


PATH = 'C:/Users/Marie Bienvenu/stage_m2/paper/'
assert os.path.exists(PATH), f"Directory not found: {PATH}"

for filename in os.listdir(PATH):

    name, extension = filename.split(".")

    if extension == 'html':

        fig = read_from_html(PATH+filename)
        fig.update_layout(xaxis=None, yaxis=None, title=None)
        fig.show()
