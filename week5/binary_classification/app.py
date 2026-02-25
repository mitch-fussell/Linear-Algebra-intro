import torch
import torch.nn as nn
import gradio as gr

model_data = torch.load('model.pth')

fm = model_data['fm']
fs = model_data['fs']
parameters = model_data['parameters']

linear = nn.Linear(1,1)
linear.load_state_dict(parameters)

model = nn.Sequential(
    linear,
    nn.Sigmoid()
)

def f(tumor_size):
    features = torch.tensor([
        [tumor_size]
    ]).float()

    #standardizing
    X = (features - fm)/fs

    classification = model(X)
    
    if classification >= 0.5:
        result = 'Malignant'
    else: 
        result = 'Benign'
    
    return result



with gr.Blocks() as iface:
    tumor_size = gr.Number(label = 'Provide the tumor size')
    diagnosis_box = gr.Text(label = 'Diagnosis prediction')
    
    tumor_size.change(fn = f, inputs = [tumor_size], outputs = [diagnosis_box])
    
    
iface.launch()