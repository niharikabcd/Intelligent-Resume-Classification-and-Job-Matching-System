import gradio as gr
from modules.parse_pdf import process_pdf
from modules.classify import classify_text_multi


def process_and_classify_pdf(file):
    # Process the PDF to extract and clean the text
    parsed_text = process_pdf(file)
    # Classify the parsed text
    classification = classify_text_multi(parsed_text)
    # Return both parsed text and classification result
    return parsed_text, classification


# Define Gradio interface
input_file = gr.File(label="Upload PDF")
output_text = gr.Textbox(label="Parsed Text")
output_class = gr.Textbox(label="Ideal Job Title(s)")

gr.Interface(
    fn=process_and_classify_pdf,
    inputs=input_file,
    outputs=[output_text, output_class],
    title="Resume Classification and Parsing for Intelligent Applicant Screening",
    theme=gr.themes.Soft()
).launch()