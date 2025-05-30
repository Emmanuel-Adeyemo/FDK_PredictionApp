import dash
from dash import dcc, html

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pages.welcome as welcome
import pages.train_val as train_val
import pages.prediction as prediction

from components.footer import footer

font_awesome = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.SLATE, font_awesome])
server = app.server

app.title = 'FDK Prediction App'

# Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    dbc.NavbarSimple(children=[
        dbc.NavItem(dbc.NavLink('Training & Validation Insights', href='/train')),
        dbc.NavItem(dbc.NavLink('Prediction', href='/predict'))
    ], brand=html.A('FDK Prediction App', href='/', style={'textDecoration': 'none', 'color': 'white'}),
        color='primary', dark=True),
    html.Div(id='page-content'),

    footer

])


# Page Nav callbacks
@dash.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/train':
        return train_val.layout
    elif pathname == '/predict':
        return prediction.layout
    else:
        return welcome.layout


if __name__ == '__main__':
    app.run(debug=True)
