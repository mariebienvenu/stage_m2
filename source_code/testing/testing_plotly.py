
import app.visualisation as vis
import plotly
import kaleido

print(plotly.__version__, kaleido.__version__)

fig = vis.add_curve(y=[1,3,4,2,0])

fig.write_image("C:/Users/Marie Bienvenu/stage_m2/afac/1.png")  #, height=1080, width=1920, scale=2, validate=True)
fig.show()


vis.add_curve(y=[1,3,2], linewidth=3).show()
