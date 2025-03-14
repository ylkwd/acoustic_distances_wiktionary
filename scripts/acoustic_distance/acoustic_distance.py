#!/usr/bin/env python
import os
import dtw
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

def read_data(base_path, model):
    features = {}
    model_parts = model.split('/')
    if len(model_parts) != 2:
        raise ValueError("Model format should be 'directory/layer'")
    
    model_dir, layer_str = model_parts
    pattern = os.path.join(base_path, model_dir, "*", layer_str + ".npy")
    print("Loading features from:", pattern)
    files = sorted(glob(pattern))
    
    for f in tqdm(files):
        instance_folder = os.path.basename(os.path.dirname(f))
        parts = instance_folder.split('_', 1)
        instance_id = parts[0]
        label = parts[1]
        feature_array = np.load(f)
        features[instance_id] = (label, feature_array)
    
    return features

def compute_dtw_distance(features1, features2):
    results = []
   
    if len(features1.keys()) == len(features2.keys()):
        common_ids = set(features1.keys()).intersection(features2.keys())
        print(f"Computing distances...")
        print(f"{round(((len(common_ids) / len(features1.keys())) * 100) , 1)}% of words retained")
        
        for instance_id in tqdm(sorted(common_ids)):
            label1, feat1 = features1[instance_id]
            label2, feat2 = features2[instance_id]
            dobj = dtw.dtw(feat1, feat2)
            dtw_distance = dobj.normalizedDistance
            results.append({
                "instance_id": instance_id, 
                "label_gpt4o": label1, 
                "label_wiktionary": label2, 
                "dtw_distance": round(dtw_distance, 4)
            })
        
        df = pd.DataFrame(results)
        return df

def main():
    model = 'wav2vec2-large-960h/layer-10'
    
    base_dir = "../../feats/wiktionary_pronunciations-final"
    gpt4o_path = os.path.join(base_dir, "GPT4o")
    wiktionary_path = os.path.join(base_dir, "wiktionary")
    
    features_gpt4o = read_data(gpt4o_path, model)
    features_wiktionary = read_data(wiktionary_path, model)
    
    dtw_df = compute_dtw_distance(features_gpt4o, features_wiktionary)
    
    output_dir = "../../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "dtw_distances_" + model.split("/")[-1] + ".csv")
    dtw_df.to_csv(output_file, index=False)
    print("Done! Computed DTW distances for {} instances. Results saved to: {}".format(len(dtw_df), output_file))

if __name__ == '__main__':
    main()
