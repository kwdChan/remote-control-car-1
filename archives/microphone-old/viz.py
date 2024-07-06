from typing import List
import numpy as np
import plotly.graph_objects as go



def get_running_spectrogram(freqs, total_frames=200):


    data = [np.zeros(len(freqs)) for _ in range(total_frames)]
    heatmap_fig = go.FigureWidget([go.Heatmap( y = freqs, zmin=0, zmax=6)])


    def update_figure(sxx_frames):
        data.pop(0)
        data.append(sxx_frames)
    
        trace = heatmap_fig.data[0]
        trace.z = np.stack(data, axis=-1) # type: ignore 

    return heatmap_fig, update_figure

def get_power_plot(freqs):

    line_fig = go.FigureWidget([go.Scatter( x = freqs)])

    def update_figure(value):
        line_fig.data[0].y = value # type: ignore 

    return line_fig, update_figure

def get_table(columns: List[str]):
    fig = go.FigureWidget(go.Table(
        header=dict(values=columns),
        cells=dict(values=[[], []])
    ))

    def update_table(data_dict):
        """
        take 


        fig, update_table  = get_updating_table(['a', 'b'])
        update_table({'a':[1,2,3], 'b':[2,4,5]})
        """
        values = []
        for c in columns:
            values.append(data_dict[c])
        
        fig.data[0].cells.values = values # type: ignore 
    return fig, update_table