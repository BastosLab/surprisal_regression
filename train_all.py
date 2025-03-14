#!/usr/bin/env python

import glob
import hdf5storage as mat
import itertools
import json
import os
import re
import subprocess
from tqdm import tqdm

ABLATIONS = ["repetition", "surprise"]
ABLATION_SETTINGS = list(itertools.chain(
    itertools.combinations(ABLATIONS, 0), itertools.combinations(ABLATIONS, 1),
    itertools.combinations(ABLATIONS, 2)
))
AREAS_HIERARCHY = ['PFC', 'FEF', 'MST', 'MT', 'V4', 'V3', 'V2', 'V1']

LOGDIR_RE = r"surprisal_regression/logs/train/runs/(.*)/checkpoints/"

SESSION_DIR = '/mnt/data/surprisal_coding/epoched'
SESSIONS = sorted(glob.glob(SESSION_DIR + "/*.mat"))
TRAINING_SESSIONS = {}

def session_areas(session):
    raw = mat.loadmat(session, squeeze_me=True)['datastruct']
    raw = dict(zip(raw.dtype.names, raw.item()))
    possible_areas = [area.item() for area in raw['areas'][:, 0].tolist()]
    timestamps = {
            k: v for (k, v) in zip(possible_areas, raw['times_in_trial'][0, :].tolist())
            if len(v)
    }
    del raw
    for area in timestamps.keys():
        if area in AREAS_HIERARCHY:
            yield area

for session in tqdm(SESSIONS, desc='Sessions'):
    session_spec = 'data.session_path=%s' % session
    TRAINING_SESSIONS[session] = {}
    for area in tqdm(session_areas(session), desc='Areas', leave=False):
        area_spec = 'data.area=%s' % area
        TRAINING_SESSIONS[session][area] = []
        for ablations in tqdm(ABLATION_SETTINGS, desc='Ablations', leave=False):
            ablation_spec = 'model.importance.ablations=%s' % str(list(ablations))
            completed = subprocess.run(['python', 'src/train.py', session_spec,
                                        area_spec, ablation_spec],
                                       stderr=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL)
            m = re.search(LOGDIR_RE, completed.stderr.decode('utf-8'))
            if m is not None:
                m = m.groups()[0]
            TRAINING_SESSIONS[session][area].append((ablations, m))

with open("TRAINING_SESSIONS.json", "w") as f:
    json.dump(TRAINING_SESSIONS, f)
