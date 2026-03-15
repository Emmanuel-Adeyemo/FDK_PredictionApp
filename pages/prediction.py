import io

import dash
import dash_table
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b2
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from PIL import Image
from pathlib import Path
import base64
from decimal import Decimal
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

model_path = 'models/model_b2_aug_LRscheduler_WD_MAE.pth'

device = torch.device('cpu')
model = efficientnet_b2(weights=None)
# regression layer
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def predict_image(image):
    image_tens = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_score = model(image_tens).item() * 100

    return predicted_score


def remove_background(image):
    # img = cv2.imread(image_path)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


def generate_gradcam(model, original_pil_img, img_tensor):
    img_float = np.float32(original_pil_img) / 255.0

    target_layers = [model.features[7]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
    grayscale_cam_resized = cv2.resize(grayscale_cam, (img_float.shape[1], img_float.shape[0]))

    visualization = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=False)

    return visualization



def process_uploaded_images(img_contents, filenames):
    predicted_data = []
    for i in range(len(img_contents)):
        content_type, content_string = img_contents[i].split(',')
        img_binary = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(img_binary)).convert("RGB")

        predicted_score = predict_image(image)
        display_score = round(predicted_score)

        predicted_data.append({
            'image_id': Path(filenames[i]).stem,
            'predicted_score_multiple': display_score
        })
    return pd.DataFrame(predicted_data)


def process_true_scores_csv(csv_contents, csv_filename, df_predicted):
    df_merged = df_predicted.copy()
    df_merged['true_score'] = 'N/A'
    df_merged['residual_score'] = 'N/A'

    if csv_contents:
        try:
            content_type, content_string = csv_contents.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            temp_df = pd.read_csv(io.StringIO(decoded))

            if 'image_id' in temp_df.columns and 'true_score' in temp_df.columns:
                true_scores_df = temp_df[['image_id', 'true_score']].copy()
                true_scores_df['image_id'] = true_scores_df['image_id'].astype(str)

                df_merged = pd.merge(df_predicted, true_scores_df, on='image_id', how='left')
                df_merged['true_score'] = df_merged['true_score'].fillna('N/A')

                df_merged['true_score_for_calc'] = pd.to_numeric(df_merged['true_score'], errors='coerce')
                df_merged['residual_score'] = np.where(
                    df_merged['true_score_for_calc'].notna(),
                    round(df_merged['predicted_score_multiple'] - df_merged['true_score_for_calc'], 2),
                    'N/A'
                )
                df_merged = df_merged.drop(columns=['true_score_for_calc'])
            else:
                print(
                    "CSV must contain 'image_id' and 'true_score' columns. Keeping N/A for true scores/residuals.")
        except Exception as e:
            print(f"Error processing CSV: {e}. Keeping N/A for true scores/residuals.")
    return df_merged


def generate_plots(df_for_plots):

    df_for_plots['true_score_num_plot'] = pd.to_numeric(df_for_plots['true_score'], errors='coerce')
    df_for_plots['predicted_score_num_plot'] = pd.to_numeric(df_for_plots['predicted_score_multiple'], errors='coerce')

    df_for_plots = df_for_plots.dropna(subset=['true_score_num_plot', 'predicted_score_num_plot'])

    df_for_plots['residual_score_num_plot'] = df_for_plots['true_score_num_plot'] - df_for_plots[
        'predicted_score_num_plot']

    test_r_squared = r2_score(df_for_plots['true_score_num_plot'], df_for_plots['predicted_score_num_plot'])
    MAE_test = mean_absolute_error(df_for_plots['true_score_num_plot'], df_for_plots['predicted_score_num_plot'])

    # scatter
    scatter_fig = px.scatter(df_for_plots, x='true_score_num_plot', y='predicted_score_num_plot',
                             title='Predicted vs True Scores',
                             labels={'true_score_num_plot': 'Actual Score',
                                     'predicted_score_num_plot': 'Predicted Score'},
                             opacity=0.7, hover_data=['image_id'])

    xymin = min(min(df_for_plots['true_score_num_plot']), min(df_for_plots['predicted_score_num_plot']), 0)
    xymax = max(max(df_for_plots['true_score_num_plot']), max(df_for_plots['predicted_score_num_plot']))
    line_range = [xymin, xymax]

    scatter_fig.add_trace(
        go.Scatter(line=dict(color='#FFC107', dash='dash'), x=line_range, y=line_range, mode='lines',
                   showlegend=False))

    scatter_fig.add_annotation(
        x=10, y=max(df_for_plots['true_score_num_plot'])-10,
        text=f'R²: {test_r_squared:.2f}<br>MAE: {np.round(MAE_test, 2)}',
        showarrow=False, bgcolor='white', font=dict(color='#343a40', size=14)
    )

    scatter_fig.update_layout(
        yaxis_title='True Score',
        xaxis_title='Predicted Score',
        hovermode='closest',
        template='plotly_white',
        showlegend=False,
        font=dict(family="sans-serif", size=12, color="#495057"),
        xaxis=dict(
            showgrid=True,
            gridcolor='#f8f9fa',
            linecolor='#dee2e6'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f8f9fa',
            zeroline=False
        )
        )

    # residual
    residual_fig = go.Figure()

    residual_fig.add_trace(go.Scatter(
        x=df_for_plots['true_score_num_plot'],
        y=df_for_plots['residual_score_num_plot'],
        mode='markers',
        name='Residuals',
        marker=dict(
            color='red',
            opacity=0.6,
            line=dict(
                color='white',
                width=1
            )
        )
    ))

    residual_fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="grey",
        annotation_text="Zero Residuals",
    )

    residual_fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,30,0,0)',
        plot_bgcolor='rgba(0,30,0,0)',
        showlegend=False,
        font=dict(family="sans-serif", size=12, color="#495057"),
        xaxis=dict(
            showgrid=True,
            gridcolor='#f8f9fa',
            linecolor='#dee2e6'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f8f9fa',
            zeroline=False
        )
    )


    return scatter_fig, residual_fig


# App Layout
layout = dbc.Container([
    html.H2('Prediction', className='text-left mt-4 mb-4 '),

    dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(" Upload Limit Reached")),
            dbc.ModalBody([
                html.P("The batch of images you selected is too large for the browser to process at once.", className="fw-bold"),
                html.Ul([
                    html.Li("Why? High-resolution images create massive data strings that crash browser memory."),
                    html.Li("Limit: Please try uploading 50 images or fewer at a time."),
                    html.Li("Fix: Converting images to JPG or resizing them before upload will allow for much larger batches.")
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Got it", id="close_error_modal", className="ms-auto", n_clicks=0)
            ),
        ], id="error_modal", is_open=False, size="lg", centered=True),

    dbc.Row([
        dbc.Col([
            html.H5('Upload Images'),
            dcc.Loading(
                id='loading_image_upload',
                type='circle',
                color='#343a40',
                children=[
                    html.Div(id='hidden_image_upload_output', style={'display': 'none'})

                ]
            ),
            dcc.Upload(id='image_upload',
                       children=html.Button('Upload'),
                       multiple=True,
                       accept='.tif, .png, .jpg',
                       style={'width': '100%', 'padding': '10px', 'border': '1px dashed white'}),
            html.Div([
                html.P(id='upload_feedback', className='text-center')
            ], style={'marginTop': '10px'}),
            dbc.RadioItems(id='upload_type',
                           options=[{'label': 'Single Image', 'value': 'single'},
                                    {'label': 'Multiple Images', 'value': 'multiple'}],
                           value='single',
                           inline=True),
            dbc.Checklist(id='check_true_score',
                          options=[{'label': 'Are true scores available?', 'value': 'true_scores'}],
                          value=[],
                          inline=True),
            html.Div(id='csv_upload_section', style={'display': 'none'}, children=[
                html.H5('Upload True Scores (CSV)'),
                dcc.Loading(
                    id='loading_csv_upload',
                    type='circle',
                    color='#343a40',
                    children=[
                        html.Div(id='hidden_csv_upload_output', style={'display': 'none'})

                    ]
                ),
                dcc.Upload(id='csv_upload',
                           children=html.Button('Upload True Scores CSV'),
                           multiple=False,
                           accept='.csv',
                           style={'width': '100%', 'padding': '10px', 'border': '1px dashed white',
                                  'marginBottom': '10px'}),

                html.Div([
                    html.P(id='csv_upload_feedback', className='text-center')
                ], style={'marginTop': '10px'}),
            ]),
            dbc.Button('Run Prediction', id='predict_button', color='primary', className='mt-3'),

            dcc.Loading(
                id='loading_spinner',
                type='circle',
                color='#343a40',
                children=[html.Div(id='loading')]
            )
        ], width=6)
    ], className='mb-4'),

    # one image
    dbc.Row(
        [

            dbc.Col([
                html.Div([
                    html.Img(id='uploaded_img', style={'width': '100%', 'borderRadius': '10px'}),
                    html.P(id='predicted_score', className='text-center')
                ]),
                # html.P(id='predicted_score', className='text-center')
            ], width=3, style={'flexDirection': 'column'}),

            dbc.Col([
                html.Div([
                    html.Img(id='grad_cam_img', style={'width': '100%', 'borderRadius': '10px'}),
                    html.P('Grad-CAM Heatmap', id='heatmap_label', className='text-center',
                           style={'display': 'none'})
                ]),
            ], width=3, style={'flexDirection': 'column'}),

        ], className='mb-4', id='single_img_contain', style={'display': 'flex', 'justifyContent': 'space-between'}),


    # multiple images


    # no grad cam or uploaded image if multiple is selected
    # it's just less stuff to deal with
    # also better for speed.

    dbc.Row([
        # table part
        dbc.Col([
            dash_table.DataTable(
                id='prediction_data_table_output',
                columns=[
                    {"name": "Image ID", "id": "image_id"},
                    {"name": "Predicted Score", "id": "predicted_score_multiple"},
                    {"name": "True Score", "id": "true_score"},
                    {"name": "Residual", "id": "residual_score"}
                ],
                page_size=10,
                page_action='native',
                style_table={'overflowX': 'auto', 'margin': '20px auto', 'width': '100%',
                             'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'},
                style_header={
                    'backgroundColor': '#2c3e50',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'borderBottom': '1px solid #495057',
                    'padding': '8px'
                },
                style_data={
                    'backgroundColor': '#ffffff',
                    'color': '#2c3e50',
                    'borderBottom': '1px solid #dee2e6',
                    'padding': '8px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa',
                    }
                ],
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '100px', 'width': '100px', 'maxWidth': '180px',
                },
                style_as_list_view=True,
            )
        ], width=6),

        # Plot part
        dbc.Col([
            html.Div(id='plots_area_container', style={'display': 'none'}, children=[
                html.Label('Visualize Plots'),
                dcc.Dropdown(
                    id='plot_dropdown',
                    options=[{'label': 'Scatter Plot', 'value': 'scatter_plot'},
                             {'label': 'Residual Plot', 'value': 'residual_plot'}],
                    value='test_scatter',
                    style={'color': 'black'}
                ),
                dcc.Graph(id='display_plot_graph', style={'height': '450px'})
            ])
        ], width=6)

    ], className='mb-4', id='multiple_img_contain', style={'display': 'none'}),

    dcc.Store(id='stored_results')
])


@dash.callback(
    Output('csv_upload_section', 'style'),
    Input('check_true_score', 'value'))
def load_true_scores(check_value):
    if 'true_scores' in check_value:
        return {'display': 'block'}
    return {'display': 'none'}


@dash.callback(
    Output('error_modal', 'is_open'),
    Output('predict_button', 'disabled'),
    Input('image_upload', 'filename'),
    Input('close_error_modal', 'n_clicks'),
    State('error_modal', 'is_open'),
    prevent_initial_call=True
)
def handle_upload_limit(filenames, n_clicks, is_open):
    ctx = dash.callback_context
    trigger_id = ctx.triggered_id

    if trigger_id == 'close_error_modal':
        return False, dash.no_update

    if filenames and len(filenames) > 50:
        return True, True

    return False, False




@dash.callback(
    Output('upload_feedback', 'children'),
    Output('csv_upload_feedback', 'children'),
    Output('hidden_image_upload_output', 'children'),
    Output('hidden_csv_upload_output', 'children'),
    Input('image_upload', 'filename'),
    State('image_upload', 'filename'),
    Input('csv_upload', 'contents'),
    State('csv_upload', 'filename')
)
def give_upload_feedback(img_contents, img_filenames, csv_contents, csv_filename):
    img_feedback = 'No image uploaded.'
    hidden_img_output = ''
    if img_filenames:  # used to fix 'invalid str' bug. now it checks list of names
        img_count = len(img_filenames) if isinstance(img_filenames, list) else 1
        img_feedback = f'Uploaded {img_count} image(s).'
        hidden_img_output = 'Image upload completed.'


    csv_feedback = 'No CSV uploaded.'
    hidden_csv_output = ''  # Default for hidden output
    if csv_contents:
        csv_feedback = f'Uploaded CSV: {csv_filename}'
        hidden_csv_output = 'CSV upload completed.'

    return img_feedback, csv_feedback, hidden_img_output, hidden_csv_output


@dash.callback(
    Output('prediction_data_table_output', 'data'),
    Output('predicted_score', 'children', allow_duplicate=True),
    Output('loading', 'children'),
    Output('uploaded_img', 'src'),
    Output('grad_cam_img', 'src'),
    Output('heatmap_label', 'style'),
    Output('single_img_contain', 'style'),
    Output('multiple_img_contain', 'style'),
    Output('plots_area_container', 'style'),
    Output('display_plot_graph', 'figure'),
    Output('stored_results', 'data'),
    # Output('residual', 'figure'),
    Input('predict_button', 'n_clicks'),
    # Input('plot_dropdown', 'value'),
    State('image_upload', 'contents'),
    State('image_upload', 'filename'),
    State('upload_type', 'value'),
    State('check_true_score', 'value'),
    State('csv_upload', 'contents'),
    State('csv_upload', 'filename'),
    State('plot_dropdown', 'value'),
    prevent_initial_call=True
)
def process_and_predict(n_clicks, img_contents, filenames, upload_type, true_scores_available,
                        csv_contents, csv_filename, selected_plot):
    # trigger_id = ctx.triggered_id if ctx.triggered_id else None

    if n_clicks is None:
        return ([], '', '', None, None,
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, None)

    if not img_contents:
        # If no images, plots should be hidden regardless of dropdown
        print("ERROR: img_contents is empty or None when it shouldn't be. Returning empty outputs.")
        return ([], 'Please upload images.', 'No images to process', None, None,
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
                {'display': 'none'}, None)

    # import time
    # time.sleep(2) # simulate processing time

    if not isinstance(img_contents, list):
        img_contents = [img_contents]
        filenames = [filenames]

    df_predicted = process_uploaded_images(img_contents, filenames)

    # csv if available
    df_merged = df_predicted.copy()
    if 'true_scores' in true_scores_available:
        df_merged = process_true_scores_csv(csv_contents, csv_filename, df_predicted)

    display_plot_figure = go.Figure().update_layout(template="plotly_white", paper_bgcolor='rgba(0,30,0,0)',
                      plot_bgcolor='rgba(0,30,0,0)',
                      font=dict(family="sans-serif", size=12, color="#495057"))

    if upload_type == 'single' or len(img_contents) == 1:
        # Single image display logic
        current_image_content_type, current_image_content_string = img_contents[0].split(',')
        current_image = Image.open(io.BytesIO(base64.b64decode(current_image_content_string))).convert("RGB")

        no_bkg_image = remove_background(current_image)

        image_tensor = transform(current_image).unsqueeze(0).to(device)

        gradcam_overlay_pil = generate_gradcam(model, no_bkg_image, image_tensor)
        result_pil = Image.fromarray(gradcam_overlay_pil)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        grad_cam_b64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

        # Handle original image display (especially for .tif conversion)
        original_img_bytes = io.BytesIO()
        current_image.save(original_img_bytes, format='PNG')
        uploaded_b64 = f"data:image/png;base64,{base64.b64encode(original_img_bytes.getvalue()).decode()}"

        predicted_score_text = f"Predicted Score: {df_merged.iloc[0]['predicted_score_multiple']}"

        single_img_display = {'display': 'flex'}
        multiple_img_display = {'display': 'none'}
        table_data_for_dcc_datatable = []
        heatmap_label_style = {'display': 'block'}
        plots_area_container_style = {'display': 'none'}

        # df_merged_json = {}
    else:
        # multiple
        single_img_display = {'display': 'none'}
        multiple_img_display = {'display': 'flex'}

        predicted_score_text = "See table below for predictions"
        uploaded_b64 = None
        grad_cam_b64 = None
        table_data_for_dcc_datatable = df_merged.to_dict('records')
        heatmap_label_style = {'display': 'none'}


        if ('true_scores' in true_scores_available and 'true_score'
                in df_merged.columns and df_merged['true_score'].notna().any()):
            scatter_fig, residual_fig = generate_plots(df_merged)

            if selected_plot == 'test_scatter':
                display_plot_figure = scatter_fig
            elif selected_plot == 'residual_plot':
                display_plot_figure = residual_fig

            plots_area_container_style = {'display': 'block'}
        else:
            print("Plots not generated: True scores not available or CSV not valid/uploaded.")
            plots_area_container_style = {'display': 'none'}

    return (table_data_for_dcc_datatable,
            predicted_score_text,
            'Processing Complete',
            uploaded_b64,
            grad_cam_b64,
            heatmap_label_style,
            single_img_display,
            multiple_img_display,
            plots_area_container_style,
            display_plot_figure,
            df_merged.to_dict('records')
            )


@dash.callback(
    Output('display_plot_graph', 'figure', allow_duplicate=True),
    Input('plot_dropdown', 'value'),
    State('stored_results', 'data'),
    prevent_initial_call=True
)
def update_plot_on_dropdown_change(selected_plot, stored_data):
    if not stored_data:
        return go.Figure()

    df = pd.DataFrame(stored_data)

    scatter_fig, residual_fig = generate_plots(df)

    if selected_plot == 'scatter_plot':
        return scatter_fig
    else:
        return residual_fig
