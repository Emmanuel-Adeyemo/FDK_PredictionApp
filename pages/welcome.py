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
            html.P("This tool leverages Deep Learning to enhance efficiency in Fusarium Damaged Kernels (FDK) scoring!",
                   className="fade-in")
        ]),
        html.Hr(),
        html.Div([
            html.P(
                "1. Fusarium Head Blight (FHB) is a damaging fungal disease that affects wheat, reducing both yield and grain quality."),

            html.P(
                "2. Fusarium Damaged Kernels (FDK) are estimated as the percentage of kernels in a sample showing characteristic FHB symptoms (discoloration or shriveling)."),

            html.P(
                "3. This application utilizes a Deep Learning pipeline to automate FDK scoring, providing an objective alternative to manual visual estimation."),

            html.P(
                "4. Digital images were captured using a Nikon D300S DSLR (60mm lens) at a height of 37cm."),

            html.P(
                "5. A dataset of 801 images was used for development (80:20 split), with 156 independent images reserved for final test evaluation."),

            html.P(
                "6. To improve model generalization, the training pipeline incorporated geometric data augmentations, including random horizontal and vertical flips, ensuring the model is less sensitive to grain orientation."),

            html.P(
                "7. The architecture, based on EfficientNet-B2, was optimized using AdamW and a Learning Rate scheduler, achieving a validation MAE of 8.3."),

            html.P(
                "8. On the independent test set, the model reached performed with a Mean Absolute Error (MAE) of 6.5 and an R² of 0.66.")

        ], className='text-left'),

        dbc.Button('Get Started', href='/train', color='primary', className='mx-auto shadow-lg')
    ], className='mt-5')
])
