import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
import plotly.express as px
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import tempfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline    
from SlidingWindowTransformer import  *
from streamlit_utils import *
import torch
from peft import PeftModel, PeftConfig
from MlpClassification import *
import pickle

# ================= Streamlit SETTINGS =======================================================
st.set_page_config(page_title="Text Classification", page_icon="🩺", layout="wide")

st.title("KDL Classifier")

# KDL hierarchy
df = pd.read_csv("kdl.csv")

cls = df["display"].tolist()

# color map for all plots
palette = px.colors.qualitative.Plotly
lvl1_labels = df["lvl1_display"].unique()
#lvl1_labels = [label +  for label in lvl1_displays]

# If colour map > lvl1 classes wrap around
color_map = {lvl1_labels[i]: palette[i % len(palette)] for i in range(len(lvl1_labels))}

# Placeholder color for plotly
color_map["(?)"] = "#f0f0f0"

# ================= SIDEBART SETTINGS =======================================================
st.sidebar.markdown("# Advanced Settings")

with st.sidebar:
    overlap = st.slider("Overlap Size", 0, 512, 128, help="1")

    top_k_classes = st.slider("Display TOP n Classes", 1, len(cls), 5, help="1")
    
    
    models_ckpt = ie = st.selectbox(
    'Model',
    ('DistilBert', 'MedBert', 'MLP'),
    help="""Choose one of the finetuned Models.
        """)

    pooling_method = ie = st.selectbox(
    'Pooling Method',
    ('mean', 'max'),
    help="""How to combine chunks
        """)
    
    export_predictions = st.toggle("Export Predictions", help="Export")


    # Export Folder + Config File
    if export_predictions:
        folder_name = st.text_input("Output Folder Name", "predictions",
                                   help=".csv is added automatically")
        
    display_predictions = st.toggle("Display Predictions", True, help="1")

    display_chunks = False

    if display_predictions:
        display_chunks = st.toggle("Display Chunks", True, help="1")

    # default device
    device = torch.device('cpu')

    if torch.cuda.is_available():
        use_gpu = st.toggle("Use GPU", True, help=f"Available GPU: {torch.cuda.get_device_name()}")
        if use_gpu:
            device = torch.device('cuda')
        
    show_shapely_values = st.toggle("Display Shapley Values", False, help="Not recommonded without a GPU")
    
    softmax_temp = st.slider("Softmax Temperature", 0.001, 2.0, 1.0, help="1")

# Include your fine-tuned checkpoints instead
models = {"DistilBert": "distilbert/distilbert-base-german-cased",
         "MedBert": "GerMedBERT/medbert-512",
         "MLP": ".pth"}

# Used for tfidf
vectorizer = None
lsa = None

# used for transformers
tokenizer = None


# Used for shapley values        
pipe = None

# ================= MAIN  =======================================================

uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
run = False
if len(uploaded_files) > 0:
    run = st.button("Run")
else:
    st.info("Please upload a PDF file to get started.") 

# ================= PDF Processing =======================================================
if run:
    if models_ckpt != "MLP": 
        model_ckpt = models[models_ckpt]

        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,
                                                                        output_attentions=True,
                                                                        num_labels=len(cls)).to(device)
        
        model.config.id2label = {i: label for i, label in enumerate(cls)}
        model.config.label2id = {label:i for i, label in enumerate(cls)}
    else: # MLP
        model = SimpleMLP(1000, 512, len(cls))
        model.eval()

        checkpoint = torch.load(models[models_ckpt], weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # use your own vectorizer
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        with open("lsa.pkl", "rb") as f:
            lsa = pickle.load(f)

    if show_shapely_values:
       pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True
            )

    if export_predictions:
        # keys and values for config.json file
        config = {
            "overlap": overlap,
            "top_k_classes": top_k_classes,
            "model": models_ckpt,
            "pooling_method": pooling_method,
            "device": str(device),
            "display_predictions": display_predictions,
            "display_chunks": display_chunks,
            "show_shapely_values": show_shapely_values,
            "softmax_temperature": softmax_temp
        }
        
        # creats folder `folder_name`_date  and saves config.json file
        folder_name = export_config_file(folder_name, config)
        


    if len(uploaded_files) > 0 and uploaded_files is not None:
        # cached for export
        predictions = []
        probs = []

        st.markdown("<h1 style='text-align: center;'>Classification Results</h1>", unsafe_allow_html=True)

        # PDF processing
        for i, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Processing PDF {i+1} of {len(uploaded_files)} ..."):
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load PDF and extract text
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.split_documents(documents)

                # Concatenate all document chunks into a single string
                document_content = " ".join([doc.page_content for doc in docs])


                # ================= Classification =======================================================
                if models_ckpt == "DistilBert" or models_ckpt == "MedBert":
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
                

                else: # MLP
                    # MLP prediction 
                    # Check: MLPClassification for more information
                    all_logits = predict_mlp(document_content, vectorizer, lsa, model)
                
                # aggregate logits from all chunks
                aggregated_logits = pooling(all_logits, pooling_method=pooling_method)
                
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
                                    kdl_df=df)
                        
                    if display_chunks and tokenizer != None:
                        expandable_chunk_prediction(out,
                                                    tokenizer,
                                                    all_logits,
                                                    color_map=color_map,
                                                    plot_name=str(i),
                                                    top_k=top_k_classes,
                                                    pipeline=pipe,
                                                    kdl_df=df)

                
                predictions.append(cls[int(aggregated_logits.argmax(-1))])
                probs.append(round(float(torch.max(torch.softmax(aggregated_logits, dim=0))), 5))
                
                # ================= EXPORT FUNCTIONALITY =======================================================

                # Save html files
                if export_predictions:                
                    html_content = export_html_summary(df_preds,
                                                    entropy,
                                                    bar_fig,
                                                    tree_fig,
                                                    styled, # custom css for the df
                                                    probs=preds,
                                                    doc_name=f"{uploaded_file.name} -",
                                                    )
                                        
                    file_name = uploaded_file.name.split(".pdf")[0]

                    with open(f"{folder_name}/{file_name}.html", 'w') as f:
                        f.write(html_content)

        # Save CSV file
        if export_predictions:
            
            export_data = pd.DataFrame({"document_name": [f.name.split(".pdf")[0] for f in uploaded_files],
                            "prediction": predictions,
                            "probability": probs})
            
            export_data.to_csv(f"{folder_name}/summary.csv", index=None)