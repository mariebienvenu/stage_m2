import plotly.express as px
df = px.data.tips()

print(df)

fig = px.density_heatmap(df, x="total_bill", y="tip")
fig.show()