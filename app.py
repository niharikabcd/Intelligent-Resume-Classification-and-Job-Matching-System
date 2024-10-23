import gradio as gr
from modules.parse_pdf import process_pdf
from modules.classify import classify_text_multi  # Assuming this is your BERT model classification
from modules.RandomForest import classify_text_rf  # Import the single-label classification
from modules.RandomForest_Multi import classify_text_rf_multi  # Import the multi-label classification

# Function to process and classify PDF using both BERT and Random Forest models
def process_and_classify_pdf(file):
    # Step 1: Process the PDF to extract and clean the text
    parsed_text = process_pdf(file)
    
    # Step 2: Classify using the existing BERT model
    classification_bert = classify_text_multi(parsed_text)  # Assuming this is multi-label BERT model
    
    # Step 3: Classify using Random Forest single-label and multi-label
    classification_rf_single = classify_text_rf(parsed_text)
    classification_rf_multi = classify_text_rf_multi(parsed_text)
    
    # Combine the results
    combined_result = (
        f"BERT Classification: {', '.join(classification_bert)}\n"
        f"Random Forest (Single-label): {classification_rf_single}\n"
        f"Random Forest (Multi-label): {', '.join(classification_rf_multi)}"
    )
    
    # Step 4: Return parsed text and combined classification results
    return parsed_text, combined_result

# Define Gradio interface
input_file = gr.File(label="Upload PDF")
output_text = gr.Textbox(label="Parsed Text")
output_class = gr.Textbox(label="Job Title Predictions")

# Launch Gradio interface
gr.Interface(
    fn=process_and_classify_pdf,
    inputs=input_file,
    outputs=[output_text, output_class],
    title="Resume Classification and Parsing for Intelligent Applicant Screening",
    theme=gr.themes.Soft()
).launch()
