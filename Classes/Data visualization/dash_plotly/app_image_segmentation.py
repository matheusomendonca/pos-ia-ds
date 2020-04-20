
# dash imports
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

# image convert
import base64
from PIL import Image
import io

# plotly
import plotly.express as px

# scientific
import numpy as np
from sklearn.cluster import KMeans

# stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# colors
colors = {'background': '#282b38',
          'text': '#a5b1cd'}

# app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# layout
app.layout = html.Div([
    html.Div([
        html.H4("KMeans segmentation"),
        html.Div([
            dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Image')
            ]),
            style={
                'width': '90%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
                })
              ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(id='choose-k',
                         options=[{'label': str(i), 'value': i} for i in range(1, 11)],
                         value=1)], style={'width': '10%',
                                           'display': 'inline-block',
                                           'vertical-align': 'top',
                                           'margin': '10px'}),
        html.Div([
            html.Button(id='submit-button',
                        n_clicks=0,
                        children='Submit')], style={'width': '20%',
                                                    'display': 'inline-block',
                                                    'vertical-align': 'top',
                                                    'margin': '10px',
                                                    'color': colors['text']})
    ]),
    html.Div([
        html.Div(id='output-image-upload-raw', style={'width': '50%', 'display': 'inline-block'}),
        html.Div(id='output-image-upload-kmeans', style={'width': '50%', 'display': 'inline-block'})
    ])
], style={'backgroundColor': colors['background'],
          'color': colors['text']})

@app.callback(Output('output-image-upload-raw', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents')])
def raw_image(n_clicks, content):
    # raw image convert
    img = stringToRGB(content)
    raw_fig = px.imshow(img)
    raw_fig.update_xaxes(showticklabels=False)
    raw_fig.update_layout(plot_bgcolor=colors['background'],
                          paper_bgcolor=colors['background'])
    raw_fig.update_yaxes(showticklabels=False)
    raw_fig.update_traces(hovertemplate=None, hoverinfo='skip')
    
    # display
    raw_plot = html.Div([html.H4("Raw image: "),
                           dcc.Graph(figure=raw_fig)])

    return raw_plot

@app.callback(Output('output-image-upload-kmeans', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('choose-k', 'value')])
def kmeans_image(n_clicks, content, k):
    # raw image convert
    img = stringToRGB(content)

    # image kmeans
    img_kmeans = segment_img(img, k)

    # kmeans fig
    fig = px.imshow(img_kmeans)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'])
    fig.update_traces(hovertemplate=None, hoverinfo='skip')
    
    # display
    kmean_plot = html.Div([html.H4("Segmented image: "),
                           dcc.Graph(figure=fig)])

    return kmean_plot

def segment_img(img, k):
    # convert to float32
    X = img.reshape((-1,3))
    X = np.float32(X)

    # fit kmeans
    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 0, n_jobs=-1)
    kmeans.fit(X)

    # parameters
    label, center = kmeans.labels_, kmeans.cluster_centers_

    # now convert back into uint8, and make original image
    center = np.uint8(center)
    img_kmeans = center[label.flatten()]
    img_kmeans = img_kmeans.reshape((img.shape))

    return img_kmeans

def stringToRGB(base64_string):
    url = base64_string.split(',')
    image = Image.open(io.BytesIO(base64.b64decode(url[-1])))
    image = image.convert('RGB')
    return np.array(image)

if __name__ == '__main__':
    app.run_server(debug=False)