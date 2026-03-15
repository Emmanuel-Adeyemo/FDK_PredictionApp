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

loss_dta = pd.read_csv('data/epoch_train_val_loss.csv')
val_dta = pd.read_csv('data/Validation_with_residuals.csv')
combined_dta = pd.read_csv('data/labels.csv')
train_dta = pd.read_csv('data/train_only.csv')
test_dta = pd.read_csv('data/Test_Predicted_Data.csv')

train_sample_count = len(train_dta['score'])
val_sample_count = len(val_dta['True_Score'])
test_sample_count = len(test_dta['True_Score'])
combined_fdk_mean = np.mean(combined_dta['score'])
train_mean_fdk_score = np.mean(train_dta['score'])
val_mean_fdk_score = np.mean(val_dta['True_Score'])
test_mean_fdk_score = np.mean(test_dta['True_Score'])
combined_sample_count = len(combined_dta['score'])

# validation_r_squared = r2_score(val_dta['True_Score'], val_dta['Predicted_Score'])
# MAE = mean_absolute_error(val_dta['True_Score'], val_dta['Predicted_Score'])
# correlation_val, _ = pearsonr(val_dta['True_Score'], val_dta['Predicted_Score'])

test_r_squared = r2_score(test_dta['True_Score'], test_dta['Predicted_Score'])
MAE_test = mean_absolute_error(test_dta['True_Score'], test_dta['Predicted_Score'])
correlation_test, _ = pearsonr(test_dta['True_Score'], test_dta['Predicted_Score'])

#  image list

test_images = [
    {'id': 'F5_350', 'true_score': 20, 'predicted_score': 15, 'image_path': 'assets/F5_350.png', 'img_out': 'F5_350'},
    {'id': 'F5_832', 'true_score': 50, 'predicted_score': 36, 'image_path': 'assets/F5_832.png', 'img_out': 'F5_832'},
    {'id': 'F5_772', 'true_score': 80, 'predicted_score': 49, 'image_path': 'assets/F5_772.png', 'img_out': 'F5_772'},
    {'id': 'F5_807', 'true_score': 25, 'predicted_score': 24, 'image_path': 'assets/F5_807.png', 'img_out': 'F5_018'},
    {'id': 'MN19038', 'true_score': 8, 'predicted_score': 8, 'image_path': 'assets/MN19038.png', 'img_out': 'MN19038'}
]

# layout0
layout = dbc.Container([
    html.H2('Training, Validation & Testing Insights', className='text-left mt-4 mb-4'),

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
                html.P(f"Total Samples: {161}"),
                html.P(f"Mean FDK Score: {29.35}")
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
                    # 'backgroundColor': '#495057',
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
                         # {'label': 'Validation Scatter Plot', 'value': 'val_scatter'},
                         {'label': 'Test Scatter Plot', 'value': 'test_scatter'},

                         {'label': 'Distribution of Test Errors', 'value': 'residual_test'},
                         {'label': 'Test Residual Plot', 'value': 'residual_plot'},],
                value='loss', style={
                    # 'backgroundColor': '#495057',
                    'color': 'black'
                }
            ),
            dcc.Graph(id='metric_plot')
        ], width=6),

    ], className='mb-4'),

    # view few selected images
    dbc.Row([
        # Left Side: Smaller, centered image
        dbc.Col([
            html.Img(id="validation_image", style={
                "width": "80%",  # Reduced from 100%
                "display": "block",
                "margin": "auto",
                "borderRadius": "2px",
                "border": "1px solid #f8f9fa"
            }),
        ], width=5, className="d-flex align-items-center"),

        # Right Side: Refined, "Quiet" Inspector
        dbc.Col([
            html.Div([
                # Smaller, elegant Title
                html.H5(id="img_out", style={
                    "fontWeight": "400",
                    "textTransform": "uppercase",
                    "letterSpacing": "3px",
                    "color": "#6c757d"
                }),

                html.Div([
                    html.Span("Actual vs Predicted", style={
                        "fontSize": "11px",
                        "fontWeight": "700",
                        "color": "#adb5bd",
                        "textTransform": "uppercase",
                        "letterSpacing": "1px"
                    }),
                    # Gauge Graph
                    dcc.Graph(id='comparison_bar', config={'displayModeBar': False}, style={"height": "80px"})
                ], className="mt-4"),

                # Navigation moved closer to the data
                html.Div([
                    dbc.Button("←", id="prev_img_btn", color="dark", outline=True, size="sm",
                               style={"border": "1px solid #dee2e6"}),
                    html.Span(id="image_counter", style={"fontSize": "14px", "color": "#6c757d", "margin": "0 20px"}),
                    dbc.Button("→", id="next_img_btn", color="dark", outline=True, size="sm",
                               style={"border": "1px solid #dee2e6"})
                ], className="mt-4", style={"display": "flex", "alignItems": "center"})

            ], style={"paddingLeft": "40px", "borderLeft": "1px solid #f8f9fa"})
        ], width=5)
    ], className='mb-5 mt-5 justify-content-center')
])



# callback

# distribution plot
@dash.callback(Output('score_distribution_plot', 'figure'), Input('score_dropdown', 'value'))
def update_distribution_plot(selected_dataset):
    fig = None

    if selected_dataset == 'train':
        fig = px.histogram(train_dta, x='score', nbins=20,
                           title=f'FDK Score Distribution ({selected_dataset.title()})',
                           color_discrete_sequence=['#007BFF'], opacity=0.6)
    elif selected_dataset == 'val':
        fig = px.histogram(val_dta, x='True_Score', nbins=10,
                           title=f'FDK Score Distribution ({selected_dataset.title()})',
                           color_discrete_sequence=['#007BFF'], opacity=0.6)
    elif selected_dataset == 'combined':
        fig = px.histogram(combined_dta, x='score', nbins=20,
                           title='FDK Score Distribution (Train+Val)',
                           color_discrete_sequence=['#007BFF'], opacity=0.6)
    elif selected_dataset == 'test':
        fig = px.histogram(test_dta, x='True_Score', nbins=10,
                           title=f'FDK Score Distribution ({selected_dataset.title()})',
                           color_discrete_sequence=['#007BFF'], opacity=0.6)
    fig.update_layout(bargap=0.1,
                      xaxis_title='True Score',
                      yaxis_title='Count',
                      hovermode='x unified',
                      template='plotly_white',
                      paper_bgcolor='rgba(0,30,0,0)',
                      plot_bgcolor='rgba(0,30,0,0)',
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

    return fig


@dash.callback(Output('metric_plot', 'figure'), Input('metric_dropdown', 'value'))
def update_metric_plot(selected_metric):
    # loss_melted = loss_dta.melt(id_vars=['Epoch'], var_name='Loss Type', value_name='Loss')

    fig = None

    if selected_metric == 'loss':
        fig = go.Figure()

        # Train Loss
        fig.add_trace(go.Scatter(
            x=loss_dta["Epoch"], y=loss_dta["Training Loss"], name="Train Loss",
            mode="lines", line=dict(color="#343A40")
        ))

        # Validation Loss
        fig.add_trace(go.Scatter(
            x=loss_dta["Epoch"], y=loss_dta["Validation Loss"], name="Validation Loss",
            mode="lines", line=dict(color="#007BFF")
        ))
        # LR
        fig.add_vline(
            x=20,
            line_width=2,
            line_dash="dot",
            line_color="#6F42C1",
            opacity=0.6
        )

        fig.add_annotation(
            x=20.5,
            y=12,
            text="LR Decrease<br>(0.001 to 0.0001)",
            showarrow=False,
            font=dict(
                color="#6F42C1",
                size=10
            ),
            align="left"
        )



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
            hovermode="x unified",
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

        # fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)


    elif selected_metric == 'test_scatter':
        fig = px.scatter(test_dta, y='True_Score', x='Predicted_Score',
                         title='Predicted vs True Disease Scores',
                         labels={'Predicted Score': 'Predicted_Score', 'True Score': 'True_Scores'},
                         opacity=0.6, hover_data=['ID'], color_discrete_sequence=['#007BFF'])

        xymin = min(min(test_dta['True_Score']), min(test_dta['Predicted_Score']), 0)
        xymax = max(max(test_dta['True_Score']), max(test_dta['Predicted_Score']))
        line_range = [xymin, xymax]

        fig.add_trace(
            go.Scatter(line=dict(color='#FFC107', dash='dash'), x=line_range, y=line_range, mode='lines',
                       showlegend=False))

        fig.add_annotation(
            x=10, y=90,
            text=f'R²: {test_r_squared:.2f}<br>MAE: {np.round(MAE_test, 2)}',
            showarrow=False, bgcolor='white', font=dict(color='#343a40', size=14)
        )

        fig.update_layout(
            yaxis_title='True Score',
            xaxis_title='Predicted Score',
            hovermode='closest',
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

        fig.update_xaxes(showgrid=False, zeroline=False)


    elif selected_metric == 'residual_plot':

        fig = go.Figure()

        # 1. Scatter plot
        fig.add_trace(go.Scatter(
            x=test_dta['True_Score'],
            y=test_dta['Residual'],
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

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="grey",
            annotation_text="Zero Residuals",
        )

        fig.update_layout(
            title='Residuals vs True Scores',
            xaxis_title='True Score',
            yaxis_title='Residual',
            template='plotly_white',
            showlegend = False,
            font = dict(family="sans-serif", size=12, color="#495057"),
            xaxis = dict(
                showgrid=True,
                gridcolor='#f8f9fa',
                linecolor='#dee2e6'
            ),
            yaxis = dict(
                showgrid=True,
                gridcolor='#f8f9fa',
                zeroline=False
            )
            )
        fig.update_xaxes(showgrid=False, zeroline=False)






    elif selected_metric == 'residual_test':

        residuals = test_dta['Residual']
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=20, marker_color='#007BFF', opacity=0.6, name='Residuals'
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
        paper_bgcolor='rgba(0,30,0,0)',
        plot_bgcolor='rgba(0,30,0,0)',
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
        ))


    return fig


# view image
@dash.callback(
    [Output('validation_image', 'src'),
     Output('img_out', 'children'),
     Output('comparison_bar', 'figure'),
     Output('image_counter', 'children')],
    [Input('prev_img_btn', 'n_clicks'),
     Input('next_img_btn', 'n_clicks')]
)
def update_image(prev_clicks, next_clicks):
    # 1. Handle None clicks at startup
    prev_clicks = prev_clicks or 0
    next_clicks = next_clicks or 0

    total_images = len(test_images)

    # 2. Calculate index (handles both buttons and initial load)
    # This logic ensures we stay within the list bounds
    current_idx = (next_clicks - prev_clicks) % total_images

    selected = test_images[current_idx]

    # 3. Create the Figure (The Gauge)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=0, line=dict(color="#f1f3f5", width=2))

    # Actual Dot
    fig.add_trace(go.Scatter(
        x=[selected['true_score']], y=[0],
        mode="markers+text", text=["ACTUAL"], textposition="top center",
        textfont=dict(size=9, color="#495057"),
        marker=dict(color="black", size=8)
    ))

    # Model Dot
    fig.add_trace(go.Scatter(
        x=[selected['predicted_score']], y=[0],
        mode="markers+text", text=["MODEL"], textposition="bottom center",
        textfont=dict(size=9, color="#007BFF"),
        marker=dict(color="#007BFF", size=10, symbol="circle-open", line=dict(width=2))
    ))

    fig.update_layout(
        margin=dict(l=5, r=5, t=25, b=25),
        height=80,
        xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # 4. Return all 4 items as a tuple (CRITICAL)
    return (
        selected['image_path'],
        f"Sample {selected['img_out']}",
        fig,
        f"{current_idx + 1} / {total_images}"
    )
# @dash.callback(
#     [Output('validation_image', 'src'),
#      Output('img_out', 'children'),
#      Output('true_score', 'children'),
#      Output('predicted_score', 'children')],
#     [Input('prev_img_btn', 'n_clicks'),
#      Input('next_img_btn', 'n_clicks')]
# )
# def update_image(prev_clicks, next_clicks):
#     total_images = len(validation_images)
#     prev_clicks = prev_clicks or 0
#     next_clicks = next_clicks or 0

    # current_idx = (prev_clicks - next_clicks) % total_images if ctx.triggered else random.randint(0, total_images - 1)
    #
    # selected_image = validation_images[current_idx]
    #
    # return selected_image[
    #     'image_path'], f'Img Id: {selected_image["img_out"]}', f'True Score: {selected_image["true_score"]}', f'Predicted Score: {selected_image["predicted_score"]}'
