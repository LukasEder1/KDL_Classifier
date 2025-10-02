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
import math


KDL_URL = "https://simplifier.net/guide/kdl-implementierungsleitfaden-2025/Hauptseite/CodeSystem-2024?version=2025"


def score_color(score_ratio: float) -> str:
    """
    Return color based on normalized score_ratio (0=bad, 0.5=medium, 1=good)
    """
    s = max(0, min(1, score_ratio))
    if s < 0.5:
        # red → yellow
        r, g, b = 255, int(510 * s), 0
    else:
        # yellow → green
        r = int(255 * (1 - (s - 0.5) * 2))
        g = int(255 - (127 * (1 - (s - 0.5) * 2)))
        b = 0
    return f"rgb({r},{g},{b})"

def metric_with_bar(label: str,
                    value: float,
                    max_value: float = 1.0,
                    reverse: bool = False,
                    min_value: float = 0.0,
                    tool_tip_text: str = ""):
    """
    Show a metric with:
    - Top-left name
    - Large numeric value
    - Colored progress bar
    - Current value bottom-left, max value bottom-right
    """
    # Normalize value to 0..1
    ratio = min(max(value / max_value, 0.0), 1.0)
    if reverse:
        ratio = 1.0 - ratio
        max_value, min_value = min_value, max_value


    color = score_color(ratio)

    # Display metric name (top-left)
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="font-weight:600; font-size:1.1em">{label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Display main numeric value
    st.markdown(f"<div style='font-size:2em; font-weight:bold'>{value:.3f}</div>", unsafe_allow_html=True)

    # Display progress bar with current/max values
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; width:100%; margin-bottom:10px;">
    <div style="height:10px; background:#eee; border-radius:4px; position:relative;">
        <div style="width:{ratio*100}%; height:100%; background:{color}; border-radius:4px;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.7em; color:#555;">
        <div>{min_value:.3f}</div>
        <div>{max_value:.3f}</div>
    </div>
</div>

    """, unsafe_allow_html=True)

def plot_lime_explanation(explainer):
    
    indexed_string = explainer.domain_mapper.indexed_string
    local_explanations = explainer.local_exp
    #st.markdown(f"<h3 style='text-align: center;'>LIME Explanation R^2 Score: <b>{explainer.score}</b></h3>", unsafe_allow_html=True)
    metric_with_bar("LIME R^2 Score", explainer.score, max_value=1.0, min_value=0.0, reverse=False, tool_tip_text="R^2 Score indicates how well the linear model fits the prediction. A score of 1.0 means perfect fit.")
    for label in explainer.top_labels:
        words = []
        values = []
        for idx, lime_value in local_explanations[label]:
            words.append(str(indexed_string.word(idx)))
            
            values.append(lime_value)
        
        df = pd.DataFrame({"word": words,
                            "value": values})
        
        #df = df.set_index("word").loc[words].reset_index()
        df["Influence"] = df["value"].apply(lambda x: "Negative" if x < 0 else "Positive")
        
        fig = px.bar(df, 
                x="value",
                y="word",
                color="Influence",
                orientation='h',
                category_orders={"word": words},  # force custom order
                color_discrete_map={"Positive": "lightgreen", "Negative": "lightcoral"},

        )

        fig.update_layout(
            
            yaxis={'categoryorder':'total ascending', 'automargin': True},
            yaxis_type="category" 
            )

        fig.update_xaxes(
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2
        )
        fig.update_traces(marker=dict(cornerradius=30))
        st.markdown(f"<h3 style='text-align: center;'>LIME Explanation for class: <b>{explainer.class_names[label]}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size:1.3vw;'><span style='background-color: lightgreen;padding: 2px;margin: 1px;border-radius: 5px;'>Positive</span> tokens increase the likelihood of the class<br> <span style='background-color: lightcoral;padding: 2px;margin: 1px;border-radius: 5px;'>Negative</span> tokens decrease it.</p>", unsafe_allow_html=True)
                          
        st.plotly_chart(fig)

        all_words = [str(word) for word in indexed_string.inverse_vocab]

        html = """<div style='text-align: center; font-size:1.3vw;'>"""

        for word in all_words:
            # TODO find better fix
            word = word.strip()
            if word in df["word"].values:
                current = df[df["word"] == word]
                color = current["Influence"].values[0]
                value = current["value"].values[0]
                
                word = (
                        f"<span class='{color}'>"
                        f"<b>{word}</b>"
                        f"<span class='value'> {round(value, 3)}</span>"
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

    header_style = {
        "selector": "th",
        "props": [("background-color", "#696969"),  
                  ("color", "white"),
                  ("font-weight", "bold"),
                  ("text-align", "center")]
    }
    # Create a Styler, style the first column (Level 1) via applymap, and hide the index.
    styled = (
        df[["Level 1", "Prediction", "Probability (%)","Description"]].style
        #.set_properties(subset=["Prediction"], **{"font-size": "30px"})
        # Color each Level 1 cell via our color_by_Level 1 function
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
                custom_data=["Probability (%)", "Level 1", "Parent"])
        
    fig.update_layout(barmode='stack', 
                      yaxis={'categoryorder':'total ascending', 'automargin': True}, 
                      legend_title="Level 1 Class",
                      hovermode="y unified",
                      )
    

    
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


def plot_predictions(probs, plot_name, top_k, color_map, heading_lvl=2, doc_name="", kdl_df=None, language="de"):
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

    suffix=""
    if language != "de":
        suffix = "_en"

    # Dataframe containing KDL hierarchy code + display names, Descriptionriptionription and probability of each class
    df = pd.DataFrame({
        "Prediction": (kdl_df[f"display{suffix}"].values + " (" + kdl_df["code"].values + ")").tolist(), 
        "Parent": (kdl_df[f"parent_display{suffix}"].values + " (" + kdl_df["parent_code"].values + ")").tolist(),
        "Level 1": (kdl_df[f"lvl1_display{suffix}"].values), # + " (" + kdl_df["lvl1_code"].values + ")").tolist(),
        "Description": kdl_df[f"definition{suffix}"].values,
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

    st.markdown(f"<p style='text-align: center; font-size:1.3vw;'><b>Top Prediction</b>: <i>{df.iloc[0].Prediction}</i></p>", unsafe_allow_html=True)
    metric_with_bar("Entropy", H, max_value=math.log(len(probs)), min_value=0.0, reverse=True, tool_tip_text="Entropy measures the uncertainty in the model's predictions. Lower values indicate higher confidence.")
    
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
    
    st.plotly_chart(tree_fig)
    
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

    if len(doc_name) > 0:
        return df, H, bar_fig, tree_fig, styled

def expandable_chunk_prediction(txt, tokenizer, logits, plot_name, color_map, top_k, kdl_df, language, pipeline=None):
    """
    Creates expandable text sections for each chunk .

    Args:
        txt: Document string
        tokinzer: chunk SEP token (i.e.: </s>, [SEP], ...) given by the corresponding Tokenizer
        logits: unnormalized model output
        cls: List of class names
        plot_name: Name of the plot (needed for streamlit internally)
        top_k: Top k classes to display
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
    
    # 
    if len(chunks) > 1:
        for i, chunk in enumerate(chunks):
            with st.expander(chunk):
                probs = torch.softmax(logits[i].to("cpu"), dim=0)*100
               
                plot_predictions(probs,
                                f"expand_{plot_name}_{i}",
                                top_k,
                                color_map=color_map,heading_lvl=3, 
                                kdl_df=kdl_df,
                                language=language)

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