from dash import html
import dash_bootstrap_components as dbc

footer = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([html.Hr([], className='hr-footer')], width=12)
        ]),
        dbc.Row([

            dbc.Col(
                html.Div(["Built with Dash"], className='text-right'),
                width={"size": 4, "order": "last", "offset": 0},
                md={"size": 4, "order": "first"}
            ),
            dbc.Col(width=2),

            dbc.Col(
                html.Ul([
                    html.Li(
                        html.A(html.I(className="fa-brands fa-github me-3 fa-1x"), href="https://github.com/Emmanuel"
                                                                                        "-Adeyemo")
                    ),
                    html.Li(
                        html.A(html.I(className="fa-brands fa-linkedin me-3 fa-1x"), href="https://www.linkedin.com"
                                                                                          "/in/emmanuel-adeyemo-486972a2/")
                    )
                ], className='list-unstyled d-flex justify-content-center justify-content-md-end'),
                width={"size": 6, "order": "first"},
                md={"size": 6, "order": "last"}
            )
        ], className='align-items-center g-0')
    ], fluid=True)
], className='footer')