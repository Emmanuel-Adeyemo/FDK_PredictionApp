import dash
from dash import dcc, html, dash_table, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# make into four sections
# 1. dataset overview
# 2. compare distribution
# 3. validation performance
# 4. view validation image

loss_dta = pd.read_csv('data/epoch_results_retrain.csv')
val_dta = pd.read_csv('data/Validation_with_residuals_retrain.csv')
combined_dta = pd.read_csv('data/labels.csv')
train_dta = pd.read_csv('data/train_only.csv')

train_sample_count = len(train_dta['score'])
val_sample_count = len(val_dta['True_Score'])
combined_fdk_mean = np.mean(combined_dta['score'])
train_mean_fdk_score = np.mean(train_dta['score'])
val_mean_fdk_score = np.mean(val_dta['True_Score'])
combined_sample_count = len(combined_dta['score'])

validation_r_squared = r2_score(val_dta['True_Score'], val_dta['Predicted_Score'])
MAE = mean_absolute_error(val_dta['True_Score'], val_dta['Predicted_Score'])

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
        ], style={'width': '21rem'})),

        dbc.Col(dbc.Card([
            dbc.CardHeader('Validation Dataset'),
            dbc.CardBody([
                html.P(f"Total Samples: {val_sample_count}"),
                html.P(f"Mean FDK Score: {val_mean_fdk_score:.2f}")
            ])
        ], style={'width': '21rem'})),

        dbc.Col(dbc.Card([
            dbc.CardHeader('Combined Dataset'),
            dbc.CardBody([
                html.P(f'Total Samples: {combined_sample_count}'),
                html.P(f'Mean FDK Score: {combined_fdk_mean:.2f}')
            ])
        ], style={'width': '21rem'}))
    ], className='mb-4'),

    # scores histogram
    dbc.Row([
        dbc.Col([
            html.Label('Select Dataset:'),
            dcc.Dropdown(
                id='score_dropdown',
                options=[{'label': 'Training', 'value': 'train'},
                         {'label': 'Validation', 'value': 'val'},
                         {'label': 'Combined', 'value': 'combined'}],
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
                         {'label': 'Residual Plot', 'value': 'residual'}],
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
        fig = px.histogram(val_dta, x='True_Score', nbins=20,
                           title=f'FDK Score Distribution ({selected_dataset.title()})',
                           color_discrete_sequence=['#7952B3'])
    elif selected_dataset == 'combined':
        fig = px.histogram(combined_dta, x='score', nbins=20,
                           title=f'FDK Score Distribution ({selected_dataset.title()}',
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
        fig = px.line(loss_melted, x='Epoch', y='Loss', color='Loss Type',
                      title='Training vs Validation Loss Curve',
                      labels={'Epoch': 'Epochs', 'Loss': 'Loss Value'},
                      line_shape='spline',
                      color_discrete_map={'Training Loss': '#7952B3', 'Validation Loss': '#FFC107'})

        fig.update_layout(
            xaxis_title='Epochs',
            yaxis_title='Loss',
            hovermode='x unified')

    elif selected_metric == 'val_scatter':
        true_scores = np.array(val_dta['True_Score'])
        predicted_scores = np.array(val_dta['Predicted_Score'])

        # Pass linear model
        regressor = LinearRegression()
        regressor.fit(true_scores.reshape(-1, 1), predicted_scores)
        regression_line = regressor.predict(true_scores.reshape(-1, 1))

        fig = px.scatter(val_dta, x='True_Score', y='Predicted_Score',
                         title='True vs Predicted Scores',
                         labels={'True Score': 'True_Scores', 'Predicted Score': 'Predicted_Score'},
                         opacity=0.7, hover_data=['ID'])

        fig.update_traces(marker=dict(color='green'))

        fig.add_scatter(x=val_dta['True_Score'], y=regression_line,
                        mode='lines', line=dict(color='#FFC107'))

        fig.add_annotation(
            x=np.mean(val_dta['True_Score']), y=max(val_dta['Predicted_Score']),
            text=f'R²: {validation_r_squared:.2f}, MAE: {MAE:.2f}',
            showarrow=False, bgcolor='#272b30', font=dict(color='white', size=14)
        )

        fig.update_layout(
            xaxis_title='True Score',
            yaxis_title='Predicted Score',
            hovermode='closest',
            showlegend=False)

    elif selected_metric == 'residual':

        # Create interactive residual scatter plot
        fig = px.scatter(
            val_dta, x="True_Score", y="Residual",
            title="Residuals vs True Scores",
            labels={"True Score": "True_Scores", "Residual": "Residual"},
            color_discrete_sequence=['green'],
            opacity=0.7, hover_data=['ID']
        )

        # residual reference = 0
        fig.add_shape(
            type="line", x0=min(val_dta["True_Score"]), x1=max(val_dta["True_Score"]),
            y0=0, y1=0, line=dict(color='#FFC107', dash="dash"), name="Zero Residual Line"
        )

        fig.update_layout(
            xaxis_title='True Score',
            yaxis_title='Residual',
            hovermode='closest',
            showlegend=False)

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
