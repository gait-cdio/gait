import os
import yaml
import numpy as np

with open(os.path.join("output-data", 'validation-scores.yaml'), 'rb') as f:
    scores = yaml.load(f)

# TODO(rolf): compute and output mean values
aggregate_score = {}
counter = {}
for method in scores.keys():
    print(method + ":")
    aggregate_score[method]={}
    counter[method]={}
    videos=scores[method]
    for video in videos.keys():
        evals = videos[video]
        aggregate_score
        n_evals = 0
        for eval in evals.keys():
            if eval in aggregate_score[method]:
                aggregate_score[method][eval] = aggregate_score[method][eval] + evals[eval]
            else:
                aggregate_score[method][eval] = evals[eval]
            if eval in counter[method]:
                counter[method][eval] += 1
            else:
                counter[method][eval] = 1
    for eval in counter[method].keys():
        aggregate_score[method][eval] /= counter[method][eval]
        print(eval + ': ' + str(aggregate_score[method][eval]))