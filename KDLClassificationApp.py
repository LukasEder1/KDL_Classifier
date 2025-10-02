import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import tempfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline    
from SlidingWindowTransformer import  *
from streamlit_utils import *
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import csv
import shap
from peft import PeftModel, PeftConfig
from MlpClassification import *
import pickle
from lime.lime_text import LimeTextExplainer
import os

#from lime.lime_text import LimeTextExplainer

# ================= Streamlit SETTINGS =======================================================
st.set_page_config(page_title="Text Classification", page_icon="🩺", layout="wide")

st.title("KDL Classifier")

# KDL hierarchy
df = pd.read_csv("kdl.csv")
cls = df["display"].tolist()
st.markdown("""
<style>
span.Positive {
    background-color: lightgreen;
    padding: 2px;
    margin: 1px;
    border-radius: 5px;
}

span.Negative {
    background-color: lightcoral;
    padding: 2px;
    margin: 1px;
    border-radius: 5px;
}

span.value {
    margin-left: 2px;
    padding-left: 5px;
    padding-right: 3px;
    opacity: 0.5;
    text-align: right;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBART SETTINGS =======================================================
st.sidebar.markdown("# Advanced Settings")

with st.sidebar:
    language = st.selectbox(
    'Language',
    ('de', 'en'),
    help="""Choose the language for the KDL hierarchy
        """)
    #n_chunks = st.slider("Maximum #Chunks", 1, 16, 3)
    overlap = st.slider("Overlap Size", 0, 512, 128, help="1")

    top_k_classes = st.slider("Display TOP n Classes", 1, len(cls), 5, help="1")
    
    models_ckpt = st.selectbox(
    'Model',
    ('DistilBert', 'MedBert'),
    help="""Choose one of the finetuned Models.
        """)

    pooling_method = st.selectbox(
    'Pooling Method',
    ('mean', 'max'),
    help="""How to combine chunks
        """)
    
    display_predictions = st.toggle("Display Predictions", True, help="1")

    display_chunks = False

    if display_predictions:
        display_chunks = st.toggle("Display Chunks", True, help="1")

    device = torch.device('cpu')

    if torch.cuda.is_available():
        use_gpu = st.toggle("Use GPU", True, help="")#, help=f"Available GPU: {torch.cuda.get_device_name()}")
        if use_gpu:
            device = torch.device('cuda')
        
    #show_shapely_values = st.toggle("Display Shapley Values", False, help="Not recommonded without a GPU")
    

    show_lime_values = st.toggle("Display Lime Values", False, help="Not recommonded without a GPU")
    if show_lime_values:
        with st.expander("LIME Explanation", True):
            use_tokens_as_lime_features = st.toggle("Use Tokens from Tokenizer as LIME Features", False, help="Otherwise use tokens from the tokenizer")
            num_lime_features = st.slider("Number of LIME Features", 1, 10, 10, help="TOP n features to display")
            num_lime_samples = st.slider("Number of LIME Samples", 1, 6, 3, help="Number of samples to generate for LIME")
            num_classes_lime = st.slider("Number of Classes to explain with LIME", 1, 3, 1, help="TOP n classes to explain with LIME")
    #softmax_temp = st.slider("Softmax Temperature", 0.001, 2.0, 1.0, help="1")
    softmax_temp = 1.0
# Include your fine-tuned checkpoints instead
models = {"DistilBert": "Luggi/distilbert-base-german-cased-finetuned-grasc",
         "MedBert": "Luggi/medbert-512-finetuned-grasc",
        }

tokenizer = None

# ================= MAIN  =======================================================

uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
run = False
if len(uploaded_files) > 0:
    run = st.button("Run")
else:
    st.info("Please upload a PDF file to get started.") 
        
pipe = None

# ================= PDF Processing =======================================================
if run:

            # color map for all plots
    palette = px.colors.qualitative.Plotly
    lvl1_labels = df["lvl1_display"].unique()
    #lvl1_labels = [label +  for label in lvl1_displays]

    if language == "en":
   
        cls = df["display_en"].tolist()
        lvl1_labels = df["lvl1_display_en"].unique()
    
    color_map = {lvl1_labels[i]: palette[i % len(palette)] for i in range(len(lvl1_labels))}

    # Placeholder color for plotly
    color_map["(?)"] = "#f0f0f0"

    model_ckpt = models[models_ckpt]

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,
                                                                    output_attentions=True,
                                                                    num_labels=len(cls)).to(device)
    
    model.config.id2label = {i: label for i, label in enumerate(cls)}
    model.config.label2id = {label:i for i, label in enumerate(cls)}
        
    if len(uploaded_files) > 0 and uploaded_files is not None:
        predictions = []
        probs = []
        st.markdown("<h1 style='text-align: center;'>Classification Results</h1>", unsafe_allow_html=True)

        for i, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Processing PDF {i+1} of {len(uploaded_files)} ..."):
                
                  # Read in PDF Document
                document_content = read_pdf(uploaded_file)



                # ================= Classification =======================================================
                inputs = tokenize_into_chunks([document_content], 
                                    tokenizer,
                                    overlap=overlap,
                                    )
            
                input_ids = torch.tensor(inputs["input_ids"]).to(device)
                attention_mask = torch.tensor(inputs["attention_mask"]).to(device)

                out = tokenizer.decode(inputs["input_ids"][0])

                # Sliding window prediction 
                # Check: SlidingWindowTransformer for more information
                all_logits, all_attentions = predict(model, input_ids, attention_mask, 512)
            
                # aggregate logits from all chunks
                aggregated_logits = pooling(all_logits, pooling_method=pooling_method)
                #aggregated_logits = all_logits[0]
                # ================= Visualization =======================================================


                if display_predictions:
                    # Normalize logits and convert them from decimal to percent
                    preds = torch.softmax(aggregated_logits.to("cpu") / softmax_temp, dim=0)*100
                    
                    # Display all Plots and cache results for exportation
                    df_preds, entropy, bar_fig, tree_fig, styled = plot_predictions(preds,
                                    plot_name=str(i),
                                    top_k=top_k_classes,
                                    color_map=color_map,
                                    doc_name=f"{uploaded_file.name}",
                                    kdl_df=df,
                                    language=language)

                                        
                    def predict_proba(texts):
                        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
                        return probs
        
                  
                    if display_chunks and tokenizer != None:
                        expandable_chunk_prediction(out,
                                                    tokenizer,
                                                    all_logits,
                                                    color_map=color_map,
                                                    plot_name=str(i),
                                                    top_k=top_k_classes,
                                                    pipeline=pipe,
                                                    kdl_df=df,
                                                    language=language
                                                    )


                    if show_lime_values:
                        if use_tokens_as_lime_features:
                            tokens = tokenizer.tokenize(document_content)
                            document_content = "|".join(tokens)
                                    
                            lime_explainer = LimeTextExplainer(class_names=cls, split_expression=lambda x: x.split("|"))#


                        else:
                            lime_explainer = LimeTextExplainer(class_names=cls, split_expression=r'[^\w.,%/]+', bow=True)
                                                    
                        exp = lime_explainer.explain_instance(
                                document_content,  # limit to first 1000 chars for speed
                                predict_proba,
                                num_features=num_lime_features,
                                top_labels=num_classes_lime,
                                num_samples=num_lime_samples
                            )
                            
                        
                        plot_lime_explanation(exp)
                
                predictions.append(cls[int(aggregated_logits.argmax(-1))])
                probs.append(round(float(torch.max(torch.softmax(aggregated_logits, dim=0))), 5))
                
