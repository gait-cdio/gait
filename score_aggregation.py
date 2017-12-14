import os
import yaml
import numpy as np

with open(os.path.join("output-data", 'validation-scores.yaml'), 'rb') as f:
    scores = yaml.load(f)

# TODO(rolf): compute and output mean values
for method in scores.keys():
    print(method + ":", scores[method])
