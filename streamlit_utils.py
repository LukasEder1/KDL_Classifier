import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import re
import shap
import os
import json
from datetime import datetime
from pypdf import PdfReader
KDL_URL = "https://simplifier.net/guide/kdl-implementierungsleitfaden-2025/Hauptseite/CodeSystem-2024?version=2025"


def plot_lime_explanation(explainer):
    
    indexed_string = explainer.domain_mapper.indexed_string
    local_explanations = explainer.local_exp
    for label in explainer.top_labels:
        words = []
        values = []
        for idx, lime_value in local_explanations[label]:
            words.append(indexed_string.word(idx))
            values.append(lime_value)
        
        df = pd.DataFrame({"word": words,
                            "value": values})
        
        #df = df.set_index("word").loc[words].reset_index()
        df["color"] = df["value"].apply(lambda x: "Negative" if x < 0 else "Positive")
        
        fig = px.bar(df, 
                x="value",
                y="word",
                color="color",
                orientation='h',
                title=f"{label}",
                category_orders={"word": words},  # force custom order
                color_discrete_map={"Positive": "lightgreen", "Negative": "lightcoral"},

        )

        fig.update_layout(
            
            yaxis={'categoryorder':'total ascending', 'automargin': True}, 
            )

        fig.update_xaxes(
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2
        )
        fig.update_traces(marker=dict(cornerradius=30))
        st.markdown(f"<h3 style='text-align: center;'>LIME Explanation for class: <b>{label}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size:1.3vw;'><span style='background-color: lightgreen;padding: 2px;margin: 1px;border-radius: 5px;'>Positive</span> words increase the likelihood of the class<br> <span style='background-color: lightcoral;padding: 2px;margin: 1px;border-radius: 5px;'>Negative</span> words decrease it.</p>", unsafe_allow_html=True)
                          
        st.plotly_chart(fig)

        all_words = indexed_string.as_list

        html = """
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
        <div style='text-align: center; font-size:1.3vw;'>
        """
        for word in all_words:
            if word in df["word"].values:
                current = df[df["word"] == word]

                word = (
                        f"<span class='{current["color"].values[0]}'>"
                        f"<b>{word}</b>"
                        f"<span class='value'> {round(current["value"].values[0], 3)}</span>"
                        f"</span>"
                    )
            else:
                word = f"<span>{word}</span>" 
            html += word + " "

        st.markdown(html + "</div>", unsafe_allow_html=True)                           

        
        

def read_pdf(file):
    reader = PdfReader(file)

    text = []

    for page in reader.pages:
        text.append(page.extract_text())
    
    return "\n".join(text)


def export_config_file(folder_prefix, config):
    """
    Exports a config.json file consisting of all settigns used for a run.
    Additionally a folder f"{folder_prefix}_{timestamp}" is created (also contains seconds)

    Args:
        folder_prefix: Name of folder that contains all reports + config + .csv file
        config: dictonary of parameters
    """

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    folder_name = f"{folder_prefix}_{timestamp}"
    
    os.makedirs(folder_name, exist_ok=True)

    with open(f"{folder_name}/config.json", "w") as f:
        json.dump(config, f)
    
    return folder_name

def create_predictions_as_dataframe(df, color_map):
    """
    Styles the Output Distribution table

    Args:
        df: dataframe consisting of the models predictions + probs
        color_map: Dictionary of lvl1 classes and colors
    """

    def color_by_lvl1_row(row):
        """
        Returns a style dict that applies the background color to the entire row
        based on the value of the 'lvl1' column.
        """
        hex_col = color_map.get(row["Level 1"], "#FFFFFF")
        return [f"background-color: {hex_col};" for _ in row]# color: white;

    # color in th in html export
    header_style = {
        "selector": "th",
        "props": [("background-color", "#696969"),  
                  ("color", "white"),
                  ("font-weight", "bold"),
                  ("text-align", "center")]
    }

    # The css used here is mainly important for html Export
    # The only important part for streamlit is:
    #   '.apply(color_by_lvl1_row, axis=1)', which colors rows based on LVL1 classes
    styled = (
        df[["Level 1", "Prediction", "Probability (%)","Description"]].style
        .apply(color_by_lvl1_row, axis=1)
        .set_table_styles([header_style])
        .hide(axis="index")
        .format({"Probability (%)": "{:.2f} %"})
        .set_properties(subset=["Probability (%)"], **{"text-align": "center", "font-weight": "bold"})
        .set_properties(subset=["Prediction"], **{"text-align": "left", "font-weight": "bold"})
    )

    return styled

def create_bar_plot(df, color_map):     
    """
    creates the Output Distribution plot

    Args:
        df: dataframe consisting of the models Predictions + probs
        color_map: Dictionary of Level 1 classes and colors
    """

    fig = px.bar(df, 
                x="Probability (%)",
                y="Prediction",
                orientation='h', 
                color="Level 1", 
                color_discrete_map=color_map,
                custom_data=["Probability (%)", "Level 1", "Parent"] # needed for Hovermode
                )
        
    fig.update_layout(barmode='stack', 
                      yaxis={'categoryorder':'total ascending', 'automargin': True}, 
                      legend_title="Level 1 Class",
                      hovermode="y unified",
                      )
    
    # custom Hovertemplate
    fig.update_traces(marker=dict(cornerradius=30),
                    hovertemplate = "<b>Probabiltiy:</b> %{customdata[0]} %<br>" +
                "<b>Level 1 Parent: </b>%{customdata[1]}<br>" +
                "<b>Level 2 Parent:</b> %{customdata[2]}"+ "<extra></extra>")

    fig.update_yaxes(showspikes=False)
    return fig

def create_tree_map(df, color_map):
    """
    Creates the KDL hierarchy plot, based on the models output probabilities

    Args:
        df: dataframe consisting of the models Predictions + probs
        color_map: Dictionary of Level 1 classes and colors
    """

    # Hierarchical path: Level 1 -> Parent (lvl2) -> Prediction (lvl3)
    fig = px.treemap(
        df,
        path=[px.Constant("KDL Hierarchy"), 'Level 1', 'Parent', 'Prediction'],
        values='Probability (%)',
        color='Level 1',
        color_discrete_map=color_map,
        custom_data=["Probability (%)", "Level 1", "Parent", "Prediction"],
    )

        
    fig.update_layout(margin = dict(t=25, l=0, r=0, b=25),)
    
    fig.update_traces(marker=dict(cornerradius=10),
                      hovertemplate = 
                "<b>Level 3 Class: </b>%{customdata[3]}<br>" +
                "<b>Level 2 Class:</b> %{customdata[2]}<br>"+ 
                "<b>Level 1 Class:</b> %{customdata[1]}<br>"
                "<b>Probabiltiy:</b> %{customdata[0]} %" +"<extra></extra>")
    

    return fig


def plot_predictions(probs, plot_name, top_k, color_map, heading_lvl=2, doc_name="", kdl_df=None):
    """
    Prints the most likely class, plots the probability distribution of the `top_k` classes 
    and displays the corresponding dataframe

    Args:
        probs: **normalized** model output
        plot_name: Name of the plot (needed for streamlit internally)
        top_k: Top k classes to display
        color_map: Level 1 color map
        heading_lvl: Heading Hierachry of Prediction Print Statement (default h2)
        doc_name: File name printed in heading
        kdl_df: hierarchl information of each class
    """

    # Dataframe containing KDL hierarchy code + display names, Descriptionriptionription and probability of each class
    df = pd.DataFrame({
        "Prediction": (kdl_df["display"].values + " (" + kdl_df["code"].values + ")").tolist(), 
        "Parent": (kdl_df["parent_display"].values + " (" + kdl_df["parent_code"].values + ")").tolist(),
        "Level 1": (kdl_df["lvl1_display"].values), # + " (" + kdl_df["lvl1_code"].values + ")").tolist(),
        "Description": kdl_df["definition"].values,
        "Probability (%)": probs    
        }
        ).sort_values("Probability (%)", ascending=False # sorty by likelihood
        ).head(top_k) # only include top_k most likely classes
    df["Probability (%)"] = df["Probability (%)"].apply(lambda x: round(x, 2))
    
    #  Compute Entropy of Output Distribution
    # probs is given in %
    probs /= 100 
    H = -torch.sum(probs * torch.log(probs), dim=0).item()
    
    if len(doc_name) > 0:
        st.markdown(f"<h{heading_lvl} style='text-align: center;'>Results for: <b>\"{doc_name}\"</b></h{heading_lvl}>", unsafe_allow_html=True)

    st.markdown(f"<p style='text-align: center; font-size:1.3vw;'><b>Top Prediction</b>: <i>{df.iloc[0].Prediction}</i><br><b>Entropy</b>: {H:.4}</p>", unsafe_allow_html=True)
    
    # Display Output Probabilities as BAR plot
    bar_fig = create_bar_plot(df, color_map)
    st.plotly_chart(bar_fig, use_container_width=True, key=plot_name)

    # Display KDL hierarchy
    tree_fig = create_tree_map(df, color_map)
    
    st.markdown(f"<h{heading_lvl+1} style='text-align: center;'> KDL Hierarchy</h{heading_lvl+1}>", unsafe_allow_html=True)

    hierarchy = f"""<ul style='list-style-type:none;'>
                    <li><b>Level 1</b> - Klassen</li>    
                    <li><b>Level 2</b> - Unterklassen</li>
                    <li><b>Level 3</b> - Dokumenten(typ)klassen</li>
                </ul>
                """

    st.markdown(f"<div style='text-align: center; font-size:1.3vw;'><p>The corresponding code system can be found at: <a href='{KDL_URL}'>here</a>{hierarchy}</p></div>", unsafe_allow_html=True)
    
    st.plotly_chart(tree_fig, key=f"{plot_name}_tree")
    
    # Display Output Probabilities as DataFrame Table
    styled = create_predictions_as_dataframe(df, color_map)

    st.markdown(f"<h{heading_lvl+1} style='text-align: center;'> Predictions</h{heading_lvl+1}>", unsafe_allow_html=True)

    hierarchy = f"""<ul style='list-style-type:none;'>
                    <li><b>Level 1</b> - Parent Class</li>    
                    <li><b>Prediction</b> - Display Name and Code of Class</li>
                    <li><b>Probability</b> - Likelihood (%) of each Prediction</li>
                    <li><b>Description</b> - Definition of each Class</li>
                </ul>
                """

    st.markdown(f"<div style='text-align: center; font-size:1.3vw;'><p>The corresponding code system can be found at: <a href='{KDL_URL}'>here</a>{hierarchy}</p></div>", unsafe_allow_html=True)

    config = {
    "Probability (%)": st.column_config.NumberColumn("Probability (%)", format="%.2f"),
    }
    
    st.dataframe(styled, hide_index=True, column_config=config)

    # If callled by the main program, not expandable_chunk_prediction
    if len(doc_name) > 0:
        return df, H, bar_fig, tree_fig, styled

def expandable_chunk_prediction(txt, tokenizer, logits, plot_name, color_map, top_k, kdl_df, pipeline=None):
    """
    Creates expandable text sections for each chunk

    Args:
        txt: Document string
        tokenizer: used hf Tokenizer
        logits: normalized model output
        color_map: for coloring LVL1 classes
        top_k: Top k classes to display
        kdl_df: hierarchl information of each class
        pipeline: Classifaction pipeline used to compute SHAP values
    """

    # Extract [CLS] and [SEP] tokens from tokenizer
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token

    # Remove CLS token
    txt = re.sub(re.escape(cls_token), "", txt)
    # Extract All Chunks (using SEP TOKEN)
    chunks = txt.split(sep_token)[:-1]
    
    # strip whitespaces
    if len(chunks) > 1:
        chunks = [" ".join(chunk.strip().split(" ")[:-3]) for chunk in chunks]
    
    # Build shap Explainer
    if pipeline != None:
        explainer = shap.Explainer(pipeline)

        values = explainer(chunks)
    
    if len(chunks) > 1:
        for i, chunk in enumerate(chunks):
            with st.expander(chunk):
                probs = torch.softmax(logits[i].to("cpu"), dim=0)*100
               
                plot_predictions(probs,
                                f"expand_{plot_name}_{i}",
                                top_k,
                                color_map=color_map,heading_lvl=3, 
                                kdl_df=kdl_df)
                
                if pipeline != None:
                    html = shap.plots.text(values[i,:,:], display=False)
                    st.components.v1.html(html, height=1000)
            
    else:
        if pipeline != None:
            html = shap.plots.text(values[:,:,:], display=False)
            st.components.v1.html(html, height=1000)

def export_html_summary(df, H, bar_fig, tree_fig, styled, doc_name, heading_lvl=2):
    """
    Creates the html code for file {doc_name} 
    """

    html = f"""<!DOCTYPE html>
            <html>
            <head>
            <title>{doc_name.split('.pdf')[0]} Results</title>
            </head>
            <body>
        """

    heading1 = f"""<h{heading_lvl} style='text-align: center;'> 
        Prediction: {df.iloc[0].Prediction}<br>- Entropy: {H:.3f} -</h{heading_lvl}>"""

    html += heading1 

    html_table = styled.to_html()
    html += html_table

    html += "<hr>"
    html += f"""<h{heading_lvl} style='text-align: center;'>Top {len(df)} Classes Distribution</h{heading_lvl}>"""

    html += f"""<div style='width: 100%; margin: auto;'>
        {bar_fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>"""
    
    html += "<hr>"

    html += f"""<h{heading_lvl} style='text-align: center;'>KDL Hierarchy</h{heading_lvl}>"""

    # Display the figure
    html += f"""<div style='width: 100%; margin: auto;'>
        {tree_fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>"""    

    
    html += "</body></html>"
    return html