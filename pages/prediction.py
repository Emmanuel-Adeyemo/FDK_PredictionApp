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
from PIL import Image
from pathlib import Path
import base64
from decimal import Decimal
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

model_path = 'models/efficient_b2_model_retrain.pth'

device = torch.device('cpu')
model = efficientnet_b2(weights=None)
# regression layer
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[-1].in_features, 1)
)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# def fix_images_save(contents):
#     if isinstance(contents, str):
#         content_type, content_string = contents.split(',')
#         img_bits = base64.b64decode(content_string)
#         # fix and ensure RGB
#         fixed_image = Image.open(io.BytesIO(img_bits)).convert('RGB')
#     elif isinstance(contents, Image.Image):
#         fixed_image = contents.convert('RGB')
#     else:
#         # above should fix decoding problem
#         # not sure beyond this point
#         raise TypeError('Unsupported format')
#     return fixed_image

# def convert_tif2png(tif_image):
#     try:
#         img_bit = Image.open(io.BytesIO(tif_image)).convert('RGB')
#         png_buffer = io.BytesIO()
#         img_bit.save(png_buffer, format='PNG')
#
#         return png_buffer.getvalue()
#
#     except Exception as e:
#         print('Error converting .tif to .png: ', e)
#         return None

#


def predict_image(image):
    image_tens = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_score = model(image_tens).item()

    return predicted_score


def generate_gradcam(model, original_pil_img, img_tensor):
    image_uint8 = np.array(original_pil_img.convert("RGB"))
    image_float32 = image_uint8.astype(np.float32) / 255.0

    # TODO: 2 is object specific, try -1 later - final feature extraction b4 pred
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=img_tensor, targets=None)

    grayscale_cam = np.squeeze(grayscale_cam, axis=0)  # should be 7x7

    # some sanity check:
    # 1. original_pil_img to np.array(original_pil_img) to image_float32 -should be
    # original large size
    # 2. image_tensor = transform(image).unsqueeze(0).to(device) (transforms to 224x224 for
    # the model)
    # 3. cam(input_tensor=img_tensor) to grayscale_cam - made on the 224x224 input
    # show_cam_on_image function needs image_float32 to be the original image and grayscale_cam
    # to be the CAM for that original image.
    # show_cam_on_image resize grayscale_cam to match image_float32.

    target_width, target_height = original_pil_img.size
    resized_grayscale_cam = cv2.resize(grayscale_cam, (target_width, target_height))

    heatmap_overlay_cv = show_cam_on_image(image_float32, resized_grayscale_cam, use_rgb=True)

    overlayed_pil_img = Image.fromarray(heatmap_overlay_cv)

    return overlayed_pil_img


def process_uploaded_images(img_contents, filenames):
    predicted_data = []
    for i in range(len(img_contents)):
        content_type, content_string = img_contents[i].split(',')
        img_binary = base64.b64decode(content_string)
        # Ensure image is in RGB format for consistency
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

    if not df_for_plots.empty and len(df_for_plots) >= 2:  # Need at least 2 points for regression
        true_scores_array = df_for_plots['true_score_num_plot'].values.reshape(-1, 1)
        predicted_scores_array = df_for_plots['predicted_score_num_plot'].values

        try:
            regressor = LinearRegression().fit(true_scores_array, predicted_scores_array)
            regression_line = regressor.predict(true_scores_array)
            df_for_plots['regression_line'] = regression_line
            df_for_plots['residual_score_num_plot'] = df_for_plots['predicted_score_num_plot'] - df_for_plots[
                'regression_line']
        except ValueError as ve:
            print(f"Could not perform regression: {ve}. Not enough samples or issue with data.")
            df_for_plots['regression_line'] = np.nan
            df_for_plots['residual_score_num_plot'] = np.nan

        # scatter
        scatter_fig = px.scatter(df_for_plots, x='true_score_num_plot', y='predicted_score_num_plot',
                                 title='Predicted vs True Scores',
                                 labels={'true_score_num_plot': 'Actual Score',
                                         'predicted_score_num_plot': 'Predicted Score'},
                                 opacity=0.7, hover_data=['image_id'])
        if 'regression_line' in df_for_plots.columns and not df_for_plots['regression_line'].isnull().all():
            scatter_fig.add_trace(go.Scatter(x=df_for_plots['true_score_num_plot'], y=df_for_plots['regression_line'],
                                             mode='lines', name='Regression Line', line=dict(color='green')))
        scatter_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#343a40',
            plot_bgcolor='#343a40',
            font=dict(color='white'),
            xaxis=dict(gridcolor='gray'),
            yaxis=dict(gridcolor='gray')
        )

        # residual
        residual_fig = px.scatter(
            df_for_plots, x="true_score_num_plot", y="residual_score_num_plot",
            title="Residuals vs True Scores",
            labels={"true_score_num_plot": "Actual Score", "residual_score_num_plot": "Residual"},
            color_discrete_sequence=['green'],
            opacity=0.7, hover_data=['image_id'])

        residual_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#343a40',
            plot_bgcolor='#343a40',
            font=dict(color='white'),
            xaxis=dict(gridcolor='gray'),
            yaxis=dict(gridcolor='gray')
        )

    else:
        print("Not enough valid data points for regression plots.")
        scatter_fig = {}
        residual_fig = {}

    return scatter_fig, residual_fig


# App Layout
layout = dbc.Container([
    html.H2('Prediction', className='text-left mt-4 mb-4 '),

    dbc.Row([
        dbc.Col([
            html.H5('Upload Images'),
            dcc.Loading(
                id='loading_image_upload',
                type='circle',
                color='#f1f1f1',
                children=[
                    html.Div(id='hidden_image_upload_output', style={'display': 'none'})
                    # This will be the child updated by callback
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
            html.Div(id='csv_upload_section', style={'display': 'none'}, children=[  # hidden/shown
                html.H5('Upload True Scores (CSV)'),
                dcc.Loading(
                    id='loading_csv_upload',
                    type='circle',
                    color='#f1f1f1',
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
                color='#f1f1f1',
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

    # dbc.Row(
    #     [
    #         dbc.Col(html.Img(id='uploaded_img', style={'width': '80%', 'borderRadius': '10px'}), width=3),
    #         dbc.Col(html.P(id='predicted_score'), width=3),
    #         dbc.Col(html.Img(id='grad_cam_img', style={'width': '80%', 'borderRadius': '10px'}), width=3)
    #     ],
    #     id='single_img_contain'
    # ),

    # multiple images

    dbc.Row([
        dbc.Col([
            # no grad cam or uploaded image if multiple is selected
            # it's just less stuff to deal with
            # also better for speed.
            dash_table.DataTable(
                id='prediction_data_table_output',
                columns=[
                    {"name": "Image ID", "id": "image_id"},
                    {"name": "Predicted Score", "id": "predicted_score_multiple"},
                    {"name": "True Score", "id": "true_score"},
                    {"name": "Residual", "id": "residual_score"}
                ],

                style_table={'overflowX': 'auto', 'margin': '20px auto', 'width': '100%',
                             'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'},
                style_header={
                    'backgroundColor': '#343a40',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'borderBottom': '1px solid #495057',
                    'padding': '8px'
                },
                style_data={
                    'backgroundColor': '#343a40',
                    'color': '#e9ecef',
                    'borderBottom': '1px solid #495057',
                    'padding': '8px'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#495057',
                    }
                ],
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '100px', 'width': '100px', 'maxWidth': '180px',
                },
                style_as_list_view=True,
            )
        ], width=6)
    ], className='mb-4', id='multiple_img_contain', style={'display': 'none'}),

    # plots for multiple upload only
    dbc.Row(id='plots_area_container', style={'display': 'none'}, children=[
        dbc.Col([
            html.Label('Visualize Plots'),
            dcc.Dropdown(
                id='plot_dropdown',
                options=[{'label': 'Scatter Plot', 'value': 'test_scatter'},
                         {'label': 'Residual Plot', 'value': 'residual_plot'}],
                value='test_scatter', style={
                    'backgroundColor': '#495057',
                    'color': 'black'
                }
            ),
            dcc.Graph(id='display_plot_graph')
        ], width=6),

    ])
    # dcc.Store(id='stored_prediction_data')
])


@dash.callback(
    Output('csv_upload_section', 'style'),
    Input('check_true_score', 'value'))
def load_true_scores(check_value):
    if 'true_scores' in check_value:
        return {'display': 'block'}
    return {'display': 'none'}


@dash.callback(
    Output('upload_feedback', 'children'),
    Output('csv_upload_feedback', 'children'),
    Output('hidden_image_upload_output', 'children'),
    Output('hidden_csv_upload_output', 'children'),
    Input('image_upload', 'contents'),
    State('image_upload', 'filename'),
    Input('csv_upload', 'contents'),
    State('csv_upload', 'filename')
)
def give_upload_feedback(img_contents, img_filenames, csv_contents, csv_filename):
    img_feedback = 'No image uploaded.'
    hidden_img_output = ''  # Default for hidden output
    if img_contents:
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

    display_plot_figure = go.Figure().update_layout(template="plotly_dark", plot_bgcolor='#343a40',
                                                    paper_bgcolor='#343a40', font_color='white')

    if upload_type == 'single' or len(img_contents) == 1:
        # Single image display logic
        current_image_content_type, current_image_content_string = img_contents[0].split(',')
        current_image = Image.open(io.BytesIO(base64.b64decode(current_image_content_string))).convert("RGB")
        image_tensor = transform(current_image).unsqueeze(0).to(device)

        gradcam_overlay_pil = generate_gradcam(model, current_image, image_tensor)
        gradcam_bytes = io.BytesIO()
        gradcam_overlay_pil.save(gradcam_bytes, format="PNG")
        grad_cam_b64 = f"data:image/png;base64,{base64.b64encode(gradcam_bytes.getvalue()).decode()}"

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
        multiple_img_display = {'display': 'block'}

        predicted_score_text = "See table below for predictions"
        uploaded_b64 = None
        grad_cam_b64 = None
        table_data_for_dcc_datatable = df_merged.to_dict('records')
        heatmap_label_style = {'display': 'none'}

        # for dcc store
        # df_merged_json = df_merged.to_json(date_format='iso', orient='split')
        # print(df_merged)

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
            display_plot_figure
            )


@dash.callback(
    Output('display_plot_graph', 'figure', allow_duplicate=True),
    Input('plot_dropdown', 'value'),
    State('image_upload', 'contents'),
    State('image_upload', 'filename'),
    State('upload_type', 'value'),
    State('check_true_score', 'value'),
    State('csv_upload', 'contents'),
    State('csv_upload', 'filename'),
    prevent_initial_call='callback-triggered'  # only works when dropdown or button is clicked
)
def update_plot_on_dropdown_change(selected_plot, img_contents, filenames, upload_type, true_scores_available,
                                   csv_contents, csv_filename):
    # TODO: Plot reloads when dropdown is selected - fix

    if upload_type == 'single' or not img_contents:
        # If in single mode or no images, no plots should be shown
        return go.Figure().update_layout(title="No Plots in Single Mode / No Images", template="plotly_dark",
                                         plot_bgcolor='#343a40', paper_bgcolor='#343a40', font_color='white')

    if not isinstance(img_contents, list):
        img_contents = [img_contents]
        filenames = [filenames]

    df_predicted = process_uploaded_images(img_contents, filenames)
    df_merged = df_predicted.copy()

    if 'true_scores' in true_scores_available:
        df_merged = process_true_scores_csv(csv_contents, csv_filename, df_predicted)

    if ('true_scores' in true_scores_available and 'true_score' in df_merged.columns and
            pd.to_numeric(df_merged['true_score'], errors='coerce').notna().any()):
        scatter_fig, residual_fig = generate_plots(df_merged)

        if selected_plot == 'scatter_plot':
            return scatter_fig
        elif selected_plot == 'residual_plot':
            return residual_fig

    return go.Figure().update_layout(title="Not enough data for plots", template="plotly_dark", plot_bgcolor='#343a40',
                                     paper_bgcolor='#343a40', font_color='white')
