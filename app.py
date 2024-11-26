from fastapi import FastAPI
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
import threading

# Load the pre-trained model
model = load_model('my_model.keras')

# Function to make predictions
def predict(
    LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, 
    PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
):
    try:
        # Compile input features into a list
        features = [
            LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, 
            PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
            BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
            PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
        ]
        
        # Reshape input to 2D array expected by the model
        input_data = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0][0]
        
        # Convert prediction to interpretable result
        if prediction < 0.5:
            result = f"Low Risk of Default (Probability: {prediction:.2%})"
        else:
            result = f"High Risk of Default (Probability: {prediction:.2%})"
        
        return result
    
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize FastAPI app
app = FastAPI()

# Create Gradio interface with sliders and dropdowns
def create_gradio_interface():
    # Detailed sliders and inputs for each feature
    return gr.Interface(
        fn=predict,
        inputs=[
            # Credit Limit
            gr.Slider(minimum=10000, maximum=1000000, value=50000, label="LIMIT_BAL: Credit Limit"),
            
            # Sex (Categorical)
            gr.Dropdown([1, 2], value=1, label="SEX: 1=Male, 2=Female"),
            
            # Education (Categorical)
            gr.Dropdown([1, 2, 3, 4], value=1, label="EDUCATION: 1=Graduate, 2=University, 3=High School, 4=Others"),
            
            # Marriage (Categorical)
            gr.Dropdown([1, 2, 3], value=1, label="MARRIAGE: 1=Married, 2=Single, 3=Others"),
            
            # Age
            gr.Slider(minimum=18, maximum=80, value=30, label="AGE"),
            
            # Payment Status (Categorical: -2 to 8)
            gr.Slider(minimum=-2, maximum=8, value=0, label="PAY_0: Repayment Status (Current Month)"),
            gr.Slider(minimum=-2, maximum=8, value=0, label="PAY_2: Repayment Status (2 Months Ago)"),
            gr.Slider(minimum=-2, maximum=8, value=0, label="PAY_3: Repayment Status (3 Months Ago)"),
            gr.Slider(minimum=-2, maximum=8, value=0, label="PAY_4: Repayment Status (4 Months Ago)"),
            gr.Slider(minimum=-2, maximum=8, value=0, label="PAY_5: Repayment Status (5 Months Ago)"),
            gr.Slider(minimum=-2, maximum=8, value=0, label="PAY_6: Repayment Status (6 Months Ago)"),
            
            # Bill Amounts
            gr.Slider(minimum=0, maximum=500000, value=50000, label="BILL_AMT1: Bill Amount (Current Month)"),
            gr.Slider(minimum=0, maximum=500000, value=50000, label="BILL_AMT2: Bill Amount (2 Months Ago)"),
            gr.Slider(minimum=0, maximum=500000, value=50000, label="BILL_AMT3: Bill Amount (3 Months Ago)"),
            gr.Slider(minimum=0, maximum=500000, value=50000, label="BILL_AMT4: Bill Amount (4 Months Ago)"),
            gr.Slider(minimum=0, maximum=500000, value=50000, label="BILL_AMT5: Bill Amount (5 Months Ago)"),
            gr.Slider(minimum=0, maximum=500000, value=50000, label="BILL_AMT6: Bill Amount (6 Months Ago)"),
            
            # Payment Amounts
            gr.Slider(minimum=0, maximum=100000, value=10000, label="PAY_AMT1: Payment Amount (Current Month)"),
            gr.Slider(minimum=0, maximum=100000, value=10000, label="PAY_AMT2: Payment Amount (2 Months Ago)"),
            gr.Slider(minimum=0, maximum=100000, value=10000, label="PAY_AMT3: Payment Amount (3 Months Ago)"),
            gr.Slider(minimum=0, maximum=100000, value=10000, label="PAY_AMT4: Payment Amount (4 Months Ago)"),
            gr.Slider(minimum=0, maximum=100000, value=10000, label="PAY_AMT5: Payment Amount (5 Months Ago)"),
            gr.Slider(minimum=0, maximum=100000, value=10000, label="PAY_AMT6: Payment Amount (6 Months Ago)")
        ],
        outputs=gr.Textbox(label="Default Prediction"),
        title="Credit Card Default Prediction",
        description="Adjust the sliders to predict the likelihood of credit card default"
    )

# Define Gradio interface
iface = create_gradio_interface()

# FastAPI route for the home page
@app.get("/")
def home():
    return {"message": "Welcome to the Credit Default Prediction API!"}

# Function to launch Gradio interface in a separate thread to avoid blocking FastAPI
def launch_gradio():
    iface.launch(share=True, server_name="0.0.0.0", server_port=8002)

# Start Gradio in a separate thread when the FastAPI app runs
@app.on_event("startup")
def startup_event():
    # Run Gradio in a separate thread to avoid blocking FastAPI
    threading.Thread(target=launch_gradio, daemon=True).start()

# To run the app, execute this in terminal:
# uvicorn app:app --reload