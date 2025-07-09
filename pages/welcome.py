from dash import dcc, html
import dash_bootstrap_components as dbc

dbc.Container([
    html.Div(style={"backgroundColor": "#f8f9fa", "padding": "50px"}, children=[
        html.H1("Welcome to the FDK Prediction App", className="text-center"),
        html.P("AI-powered FDK scoring made efficient!", className="text-center")
    ])
])

layout = html.Div([
    dbc.Container([

        html.Div(children=[
            html.H1("Welcome!", className="text-left"),
            html.P("This tool leverages ML to enhance efficiency in Fusarium Damaged Kernels (FDK) scoring!",
                   className="fade-in")
        ]),
        html.Hr(),
        html.Div([
            html.P("1. Fusarium Head Blight (FHB) is a damaging fungal disease that affects wheat."),
            html.P("2. FDK is estimated visually as the percentage of kernels in a grain sample that are discolored due "
                   "to FHB infection."),
            html.P("3. This app provides a trained model to score FDK in digital images of hard red spring wheat "
                   "samples."),
            html.P("4. Images were captured using a Nikon D300S DSLR camera with a 60mm f/7.1 lens, positioned 37cm "
                   "above sample."),
            html.P("5. A set of 801 images were randomly split (80:20) for training and validation, "
                   "with an additional 156 images set aside for testing."),
            html.P("6. The model, based on EfficientNet B2, achieved an R² of 0.65 and MAE of "
                   "9 on validation. On the test, model reached an R² of 0.61 and MAE of 7.")

        ], className='text-left'),

        dbc.Button('Get Started', href='/train', color='primary', className='mx-auto shadow-lg')
    ], className='mt-5')
])
