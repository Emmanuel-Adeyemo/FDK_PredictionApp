import random

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, ctx
from dash.dependencies import Input, Output
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

# make into four sections
# 1. dataset overview
# 2. compare distribution
# 3. validation performance
# 4. view validation image

loss_dta = pd.read_csv('data/epoch_results_huber_7_8.csv')
val_dta = pd.read_csv('data/Validation_with_residuals_retrain_huber.csv')
combined_dta = pd.read_csv('data/labels.csv')
train_dta = pd.read_csv('data/train_only.csv')
test_dta = pd.read_csv('data/Test_Set_Predictions_Huber.csv')

train_sample_count = len(train_dta['score'])
val_sample_count = len(val_dta['True_Score'])
test_sample_count = len(test_dta['True_Score'])
combined_fdk_mean = np.mean(combined_dta['score'])
train_mean_fdk_score = np.mean(train_dta['score'])
val_mean_fdk_score = np.mean(val_dta['True_Score'])
test_mean_fdk_score = np.mean(test_dta['True_Score'])
combined_sample_count = len(combined_dta['score'])

validation_r_squared = r2_score(val_dta['True_Score'], val_dta['Predicted_Score'])
MAE = mean_absolute_error(val_dta['True_Score'], val_dta['Predicted_Score'])
correlation_val, _ = pearsonr(val_dta['True_Score'], val_dta['Predicted_Score'])

test_r_squared = r2_score(test_dta['True_Score'], test_dta['Predicted_Score'])
MAE_test = mean_absolute_error(test_dta['True_Score'], test_dta['Predicted_Score'])
correlation_test, _ = pearsonr(test_dta['True_Score'], test_dta['Predicted_Score'])

#  image list

validation_images = [
    {'id': 'F5_272', 'true_score': 30, 'predicted_score': 31, 'image_path': 'assets/F5_272.png', 'img_out': 'F5_272'},
    {'id': 'F5_640', 'true_score': 4, 'predicted_score': 10, 'image_path': 'assets/F5_640.png', 'img_out': 'F5_640'},
    {'id': 'F5_594', 'true_score': 20, 'predicted_score': 32, 'image_path': 'assets/F5_594.png', 'img_out': 'F5_594'},
    {'id': 'F5_018', 'true_score': 60, 'predicted_score': 70, 'image_path': 'assets/F5_018.png', 'img_out': 'F5_018'},
    {'id': 'F5_041', 'true_score': 90, 'predicted_score': 83, 'image_path': 'assets/F5_041.png', 'img_out': 'F5_041'}
]

# layout
layout = dbc.Container([
    html.H2('Training & Validation Insights', className='text-left mt-4 mb-4'),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('Training Dataset'),
            dbc.CardBody([
                html.P(f"Total Samples: {train_sample_count}"),
                html.P(f"Mean FDK Score: {train_mean_fdk_score:.2f}")
            ])
        ], style={'width': '16rem'})),

        dbc.Col(dbc.Card([
            dbc.CardHeader('Validation Dataset'),
            dbc.CardBody([
                html.P(f"Total Samples: {val_sample_count}"),
                html.P(f"Mean FDK Score: {val_mean_fdk_score:.2f}")
            ])
        ], style={'width': '16rem'})),

        dbc.Col(dbc.Card([
            dbc.CardHeader('Train+Val Dataset'),
            dbc.CardBody([
                html.P(f'Total Samples: {combined_sample_count}'),
                html.P(f'Mean FDK Score: {combined_fdk_mean:.2f}')
            ])
        ], style={'width': '16rem'})),

        dbc.Col(dbc.Card([
            dbc.CardHeader('Test Dataset'),
            dbc.CardBody([
                html.P(f'Total Samples: {test_sample_count}'),
                html.P(f'Mean FDK Score: {test_mean_fdk_score:.2f}')
            ])
        ], style={'width': '16rem'}))
    ], className='mb-4'),

    # scores histogram
    dbc.Row([
        dbc.Col([
            html.Label('Select Dataset:'),
            dcc.Dropdown(
                id='score_dropdown',
                options=[{'label': 'Training', 'value': 'train'},
                         {'label': 'Validation', 'value': 'val'},
                         {'label': 'Train+Val', 'value': 'combined'},
                         {'label': 'Test', 'value': 'test'}],
                value='train',
                style={
                    'backgroundColor': '#495057',
                    'color': 'black'
                }
            ),
            dcc.Graph(id='score_distribution_plot')
        ], width=6),
        # validation results
        dbc.Col([
            html.Label('Select Metric:'),
            dcc.Dropdown(
                id='metric_dropdown',
                options=[{'label': 'Loss Curve Plot', 'value': 'loss'},
                         {'label': 'Validation Scatter Plot', 'value': 'val_scatter'},
                         {'label': 'Test Scatter Plot', 'value': 'test_scatter'},
                         {'label': 'Validation Residual Plot', 'value': 'residual'},
                         {'label': 'Test Residual Plot', 'value': 'residual_test'}],
                value='loss', style={
                    'backgroundColor': '#495057',
                    'color': 'black'
                }
            ),
            dcc.Graph(id='metric_plot')
        ], width=6),

    ], className='mb-4'),

    # view few selected images
    dbc.Row([
        dbc.Col(html.Img(id="validation_image", style={"width": "100%", "borderRadius": "10px"}), width=6),
        dbc.Col([
            html.Div([
                html.P(id="img_out", style={"textAlign": "center", "fontSize": "20px"}),
                html.P(id="true_score", style={"textAlign": "center", "fontSize": "20px"}),
                html.P(id="predicted_score", style={"textAlign": "center", "fontSize": "20px"})
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "100%"
            }),
            html.Div([
                dbc.Button("Previous Image", id="prev_img_btn", color="secondary", className="me-2"),
                dbc.Button("Next Image", id="next_img_btn", color="primary")
            ], style={"display": "flex", "justifyContent": "center", 'marginBottom': '150px'})

        ], style={"display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center"},
            width=6)
    ])
], className='mt-5')


# callback

# distribution plot
@dash.callback(Output('score_distribution_plot', 'figure'), Input('score_dropdown', 'value'))
def update_distribution_plot(selected_dataset):
    fig = None

    if selected_dataset == 'train':
        fig = px.histogram(train_dta, x='score', nbins=20,
                           title=f'FDK Score Distribution ({selected_dataset.title()})',
                           color_discrete_sequence=['#7952B3'])
    elif selected_dataset == 'val':
        fig = px.histogram(val_dta, x='True_Score', nbins=10,
                           title=f'FDK Score Distribution ({selected_dataset.title()})',
                           color_discrete_sequence=['#7952B3'])
    elif selected_dataset == 'combined':
        fig = px.histogram(combined_dta, x='score', nbins=20,
                           title='FDK Score Distribution (Train+Val)',
                           color_discrete_sequence=['#7952B3'])
    elif selected_dataset == 'test':
        fig = px.histogram(test_dta, x='True_Score', nbins=10,
                           title=f'FDK Score Distribution ({selected_dataset.title()})',
                           color_discrete_sequence=['#7952B3'])
    fig.update_layout(bargap=0.1,
                      xaxis_title='True Score',
                      yaxis_title='Count',
                      hovermode='x unified',
                      template='plotly_dark',
                      paper_bgcolor='#343a40',
                      plot_bgcolor='#343a40',
                      font=dict(color='white'),
                      xaxis=dict(gridcolor='gray'),
                      yaxis=dict(gridcolor='gray')
                      )

    return fig


@dash.callback(Output('metric_plot', 'figure'), Input('metric_dropdown', 'value'))
def update_metric_plot(selected_metric):
    loss_melted = loss_dta.melt(id_vars=['Epoch'], var_name='Loss Type', value_name='Loss')

    fig = None

    if selected_metric == 'loss':
        fig = go.Figure()

        # Train Loss
        fig.add_trace(go.Scatter(
            x=loss_dta["Epoch"],
            y=loss_dta["Train Loss"],
            name="Train Loss",
            mode="lines",
            line=dict(color="green")
        ))

        # Validation Loss
        fig.add_trace(go.Scatter(
            x=loss_dta["Epoch"],
            y=loss_dta["Validation Loss"],
            name="Validation Loss",
            mode="lines",
            line=dict(color="#FFC107")
        ))
        # LR
        fig.add_trace(go.Scatter(
            x=loss_dta["Epoch"],
            y=loss_dta["Learning Rate"],
            name="Learning Rate",
            yaxis="y2",
            mode="lines",
            line=dict(color="#7952B3", dash="dash")
        ))

        fig.update_layout(
            title="Loss Curve with Learning Rate",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Learning Rate",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(x=0.6, y=0.99),
            hovermode="x unified"
        )

    elif selected_metric == 'val_scatter':

        # true_scores = np.array(val_dta['True_Score']).flatten()
        # predicted_scores = np.array(val_dta['Predicted_Score'])

        fig = px.scatter(val_dta, y='True_Score', x='Predicted_Score',
                         title='Predicted vs True Disease Scores',
                         labels={'Predicted Score': 'Predicted_Score', 'True Score': 'True_Scores'},
                         opacity=0.7, hover_data=['ID'], color_discrete_sequence=['green'])

        xymin = min(min(val_dta['True_Score']), min(val_dta['Predicted_Score']), 0)
        xymax = max(max(val_dta['True_Score']), max(val_dta['Predicted_Score']), 100)
        line_range = [xymin, xymax]

        fig.add_trace(
            go.Scatter(line=dict(color='#FFC107', dash='dash'), x=line_range, y=line_range, mode='lines',
                       showlegend=False))

        # fig.add_scatter(x=val_dta['True_Score'], y=regression_line,
        #                 mode='lines', line=dict(color='#FFC107'))

        fig.add_annotation(
            x=10, y=90,
            text=f'R²: {validation_r_squared:.2f}<br>Corr: {correlation_val:.2f}<br>MAE: {np.round(MAE)}',
            showarrow=False, bgcolor='#272b30', font=dict(color='white', size=14)
        )

        fig.update_layout(
            yaxis_title='True Score',
            xaxis_title='Predicted Score',
            hovermode='closest',
            showlegend=False)

    elif selected_metric == 'test_scatter':
        fig = px.scatter(test_dta, y='True_Score', x='Predicted_Score',
                         title='Predicted vs True Disease Scores',
                         labels={'Predicted Score': 'Predicted_Score', 'True Score': 'True_Scores'},
                         opacity=0.7, hover_data=['ID'], color_discrete_sequence=['green'])

        xymin = min(min(test_dta['True_Score']), min(test_dta['Predicted_Score']), 0)
        xymax = max(max(test_dta['True_Score']), max(test_dta['Predicted_Score']), 100)
        line_range = [xymin, xymax]

        fig.add_trace(
            go.Scatter(line=dict(color='#FFC107', dash='dash'), x=line_range, y=line_range, mode='lines',
                       showlegend=False))

        fig.add_annotation(
            x=10, y=90,
            text=f'R²: {test_r_squared:.2f}<br>Corr: {correlation_test:.2f}<br>MAE: {np.round(MAE_test)}',
            showarrow=False, bgcolor='#272b30', font=dict(color='white', size=14)
        )

        fig.update_layout(
            yaxis_title='True Score',
            xaxis_title='Predicted Score',
            hovermode='closest',
            showlegend=False)

    elif selected_metric == 'residual':

        residuals = val_dta['Residual']
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=20, marker_color='#7952B3', opacity=0.6, name='Residuals'
        ))

        fig.add_shape(
            type='line', x0=0, x1=0, y0=0, y1=1,
            line=dict(color='#FFC107', dash='dash'),
            xref='x', yref='paper'
        )

        # Update layout
        fig.update_layout(
            title="Residuals Distribution (Validataion)",
            xaxis_title="Residuals",
            yaxis_title="Frequency",
            bargap=0.1
        )

    elif selected_metric == 'residual_test':

        residuals = test_dta['Residual']
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=20, marker_color='#7952B3', opacity=0.6, name='Residuals'
        ))

        fig.add_shape(
            type='line', x0=0, x1=0, y0=0, y1=1,
            line=dict(color='#FFC107', dash='dash'),
            xref='x', yref='paper'
        )

        fig.update_layout(
            title="Residuals Distribution (Test)",
            xaxis_title="Residuals",
            yaxis_title="Frequency",
            bargap=0.1
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#343a40',
        plot_bgcolor='#343a40',
        font=dict(color='white'),
        xaxis=dict(gridcolor='gray'),
        yaxis=dict(gridcolor='gray')
    )

    return fig


# view image

@dash.callback(
    [Output('validation_image', 'src'),
     Output('img_out', 'children'),
     Output('true_score', 'children'),
     Output('predicted_score', 'children')],
    [Input('prev_img_btn', 'n_clicks'),
     Input('next_img_btn', 'n_clicks')]
)
def update_image(prev_clicks, next_clicks):
    total_images = len(validation_images)
    prev_clicks = prev_clicks or 0
    next_clicks = next_clicks or 0

    current_idx = (prev_clicks - next_clicks) % total_images if ctx.triggered else random.randint(0, total_images - 1)

    selected_image = validation_images[current_idx]

    return selected_image[
        'image_path'], f'Img Id: {selected_image["img_out"]}', f'True Score: {selected_image["true_score"]}', f'Predicted Score: {selected_image["predicted_score"]}'
