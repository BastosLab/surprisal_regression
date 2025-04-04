#!/usr/bin/env python

import functools
import glob
import hdf5storage as mat
import itertools
import json
import multiprocessing
import os
import re
import subprocess
import torch
from tqdm import tqdm

ABLATIONS = ["repetition", "surprise"]
ABLATION_SETTINGS = list(itertools.chain(
    itertools.combinations(ABLATIONS, 0), itertools.combinations(ABLATIONS, 1),
    itertools.combinations(ABLATIONS, 2)
))
AREAS_HIERARCHY = ['PFC', 'FEF', 'MST', 'MT', 'V4', 'V3', 'V2', 'V1']

LOGDIR_RE = r"surprisal_regression/logs/train/runs/(.*)/checkpoints/"
TEST_EVIDENCE_RE = r"test/log_evidence(\s)*[│](\s)*([-]?\d+[.]?\d+)"

SESSION_DIR = '/mnt/data/surprisal_coding/epoched'
SESSIONS = sorted(glob.glob(SESSION_DIR + "/*.mat"))
TRAINING_SESSIONS = {}

PARALLELIZE = True

def logmeanexp(*logws):
    return torch.tensor([*logws]).logsumexp(dim=0) - len(logws)

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

def train_ablation(session_spec, area_spec, ablations):
    ablation_spec = 'model.importance.ablations=%s' % str(list(ablations))
    completed = subprocess.run(['python', 'src/train.py', session_spec,
                                area_spec, ablation_spec],
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    m = re.search(LOGDIR_RE, completed.stderr.decode('utf-8'))
    logdir = m.groups()[0] if m is not None else None
    m = re.search(TEST_EVIDENCE_RE, completed.stdout.decode('utf-8'))
    evidence = float(m.groups()[-1]) if m is not None else None
    return (logdir, evidence)

for session in tqdm(SESSIONS, desc='Sessions'):
    session_spec = 'data.session_path=%s' % session
    TRAINING_SESSIONS[session] = {}
    for area in tqdm(session_areas(session), desc='Areas', leave=False):
        area_spec = 'data.area=%s' % area
        TRAINING_SESSIONS[session][area] = []
        if PARALLELIZE:
            with multiprocessing.Pool(4) as pool:
                train = functools.partial(train_ablation, session_spec,
                                          area_spec)
                outputs = pool.map(train, ABLATION_SETTINGS)
                for (ablations, (l, e)) in zip(ABLATION_SETTINGS, outputs):
                    TRAINING_SESSIONS[session][area].append((ablations, l, e))
        else:
            for ablations in tqdm(ABLATION_SETTINGS, desc='Ablations', leave=False):
                (logdir, evidence) = train_ablation(session_spec, area_spec,
                                                    ablations)
                TRAINING_SESSIONS[session][area].append((ablations, logdir,
                                                         evidence))
        if all(TRAINING_SESSIONS[session][area][k][2] is not None for
               k in range(len(ABLATION_SETTINGS))):
            TRAINING_SESSIONS[session][area].append(
                logmeanexp(TRAINING_SESSIONS[session][area][0][2],
                           TRAINING_SESSIONS[session][area][1][2]).item() -
                logmeanexp(TRAINING_SESSIONS[session][area][2][2],
                           TRAINING_SESSIONS[session][area][3][2]).item()
            )
        else:
            TRAINING_SESSIONS[session][area].append(None)

with open("TRAINING_SESSIONS.json", "w") as f:
    json.dump(TRAINING_SESSIONS, f, indent=4)
