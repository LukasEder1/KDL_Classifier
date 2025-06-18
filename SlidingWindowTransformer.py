import torch
import streamlit as st
# Training Function -> reused
def tokenize_into_chunks(batch, tokenizer, max_length=512, overlap=126):
    tokens = tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        stride=overlap,
        #return_tensors="pt"
    )
    # Original Document Id
    sample_mapping = tokens.pop("overflow_to_sample_mapping")

    # dataset.map enforces that number of INPUT samples == #Output samples
    # Hence we flatten the toknized batch for each key (input_ids, attention_mask)
    flattened_tokens = {k: [] for k in tokens.keys()}
    for k in tokens.keys():
        tmp_list = []
        prev = -1
        for chunk_id, doc_id in enumerate(sample_mapping):
            # if two chunks share the same original document, concat them
            if doc_id == prev:
                tmp_list[doc_id] += tokens[k][chunk_id]
            # else we are dealing with a new document
            else:
                tmp_list.append(tokens[k][chunk_id])

            # update the prev document counter
            prev = doc_id

        flattened_tokens[k] = tmp_list

    return flattened_tokens

def predict(model, input_ids, attention_mask, max_length):
    model.eval()

    
    all_chunks_input_ids = []
    all_chunks_attention_mask = []

    # Extract chunks per sample (as in compute_loss)    
    n_chunks = max(1, (input_ids[0].size(0) // max_length))

    for j in range(n_chunks):
        start = j * 512
        end = (j + 1) * 512
        chunk_input_ids = input_ids[0, start:end]
        chunk_attention_mask = attention_mask[0, start:end]

        if chunk_attention_mask.sum().item() == 0:
            continue

        all_chunks_input_ids.append(chunk_input_ids)
        all_chunks_attention_mask.append(chunk_attention_mask)
        
    # Batch all chunks and perform a single forward pass
    all_chunks_input_ids = torch.stack(all_chunks_input_ids)
    all_chunks_attention_mask = torch.stack(all_chunks_attention_mask)

    
    with torch.no_grad():
        outputs = model(input_ids=all_chunks_input_ids,
                        attention_mask=all_chunks_attention_mask)

    return outputs.logits, outputs.attentions

def sliding_window_shap_prediction(x, tokenizer, device, model, pooling_method):
    st.write("x:", x, type(x))
    inputs = tokenize_into_chunks(x, tokenizer)

    input_ids = torch.tensor(inputs["input_ids"]).to(device)
    attention_mask = torch.tensor(inputs["attention_mask"]).to(device)

    all_logits, _ = predict(model, input_ids, attention_mask, 512)

    aggregated_logits = pooling(all_logits, pooling_method=pooling_method)

    return aggregated_logits


def pooling(logits, pooling_method="mean"):
    if pooling_method == "mean":
        return torch.mean(logits, dim=0)
    elif pooling_method == "max":
        return torch.max(logits, dim=0).values
    else:
        raise ValueError(f"Unsupported Pooling method {pooling_method};\nTry max or mean pooling")
    