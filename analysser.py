#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import itertools
from CSVGenerator import CSVGenerator

FIXATION = 1
SACCADE = 0


def plot_gaze(gaze_data, gaze_labels, sid='unk', mid=0, show=False):
    data_fix = gaze_data[gaze_labels == FIXATION]
    data_sacc = gaze_data[gaze_labels == SACCADE]

    # Plot saccades
    fig = plt.figure()
    plt.scatter(data_sacc[:, 0], data_sacc[:, 1], color='b', s=0.01,
                label='Saccades')
    # Plot fixations
    plt.scatter(data_fix[:, 0], data_fix[:, 1], color='r', s=0.01,
                label='Fixations')

    plt.legend()
    plt.savefig('fig/' + sid + '_' + str(mid) + '.pdf')
    if show:
        plt.show()
    plt.close(fig)

    
def size2deg(size, distance):
    return 2 * np.degrees(np.arctan(size * .5 / distance))


def deg2size(deg, distance):
    return 2 * distance * np.tan(np.deg2rad(deg) * .5)


def units2size(units, units_range, size_range):
    units_per_size = size_range / units_range
    return units * units_per_size


def convert_data_to_deg(data):
    for values in data.values():
        for known, coords in values:
            coords[:, 0] = size2deg(units2size(coords[:, 0], 1400, 195), 450)
            coords[:, 1] = size2deg(units2size(coords[:, 1], 1400, 113), 450)

    return data


def load_data(file_name, sid_list):
    data_mat = loadmat(file_name)
    sids = data_mat['sid'].tolist()[0]
    knowns = data_mat['known'].transpose()
    coords = data_mat['coordinates'].transpose()
    
    res = {}
    for sid, known, coord in zip(sids, knowns, coords):
        vals = [known[0], coord[0]]
        if sid[0] in sid_list:
            if sid[0] not in res:
                res[sid[0]] = []
            res[sid[0]].append(vals)
    return res


def edist(a, b):
    if a.ndim == 1:
        return np.linalg.norm(a - b)
    else:
        return np.linalg.norm(a - b, axis=1)


def dur(movement_type, freq = 1000):
    durations = [] #in seconds
    for k, g in itertools.groupby(movement_type):
        g = list(g)
        if k and len(g)/freq >= 0.05: #filtering out too short fixations, probably noise
            durations.append(len(g)/ freq)
    return durations


# Mean fixation duration
def mfd(subject):
    res = [0,0,0,0,0,0]
    known = []
    unknown = []
    for row in subject:
        fixations = dur(ivt(row[1]))
        if row[0]:
            known += fixations
        else:
            unknown += fixations
    res[0] = np.mean(known)
    res[1] = np.mean(unknown)
    res[2] = np.mean(known + unknown)
    res[3] = np.std(known)
    res[4] = np.std(unknown)
    res[5] = np.std(known + unknown)
    return res, known, unknown


def get_segment_idxs(labels):
    labels_pad = np.pad(labels, (1, 0), 'edge')[:-1]
    return np.where(labels_pad != labels)[0]


def distances(coords):
    dists = edist(coords[:-1], coords[1:])
    dists = np.insert(dists, 0, 0)
    return dists


# Mean Saccade Amplitudes
def msa(subject):
    res = [0, 0, 0, 0, 0, 0]
    known = []
    unknown = []
    for row in subject:
        coords = row[1]
        mv_types = ivt(coords)
        dists = distances(coords)
        mv_types_seg = np.split(mv_types, get_segment_idxs(mv_types))
        dists_seg = np.split(dists, get_segment_idxs(mv_types))

        for mt, ds in zip(mv_types_seg, dists_seg):
            if mt[0] == 0:
                sum = np.sum(ds)
                if row[0]:
                    known.append(sum)
                else:
                    unknown.append(sum)

    res[0] = np.mean(known)
    res[1] = np.mean(unknown)
    res[2] = np.mean(known + unknown)
    res[3] = np.std(known)
    res[4] = np.std(unknown)
    res[5] = np.std(known + unknown)
    return res, known, unknown


def agg_mfd(data):
    known = []
    unknown = []
    for _, val in data.items():
        m = mfd(val)
        known += m[1]
        unknown += m[2]
    return np.mean(known), np.mean(unknown), np.std(known), np.std(unknown)


def agg_msa(data):
    known = []
    unknown = []
    for _, val in data.items():
        m = msa(val)
        known += m[1]
        unknown += m[2]
    return np.mean(known), np.mean(unknown), np.std(known), np.std(unknown)


def ivt_filter_short_fixations(gaze_labels, frequency, duration_thr=0.05):
    time_delta = 1 / frequency

    seg_idxs = get_segment_idxs(gaze_labels)
    gaze_segments = np.split(gaze_labels, seg_idxs)

    seg_idxs = np.insert(seg_idxs, 0, 0)
    seg_idxs = np.append(seg_idxs, len(gaze_labels))

    for i, seg in enumerate(gaze_segments):
        duration = len(seg) * time_delta
        if seg[0] == FIXATION and duration < duration_thr:
            # Discard the fixation
            gaze_labels[seg_idxs[i]:seg_idxs[i+1]] = SACCADE

    return gaze_labels


def ivt_trim_fix_onsets(gaze_labels, gaze_coords, dist_thr=2):
    seg_idxs = get_segment_idxs(gaze_labels)
    coords_seg = np.split(gaze_coords, seg_idxs)

    seg_idxs = np.insert(seg_idxs, 0, 0)
    seg_idxs = np.append(seg_idxs, len(gaze_labels))

    for i, coords in enumerate(coords_seg):
        labels = gaze_labels[seg_idxs[i]:seg_idxs[i+1]]
        if labels[0] == SACCADE:
            continue

        mean = np.mean(coords, axis=0)
        center_idx = edist(coords, mean).argmin()

        # Discard points too far to the left from the center
        for i in reversed(range(center_idx)):
            if edist(coords[i], coords[center_idx]) > dist_thr:
                labels[:i+1] = SACCADE
                break

        # Discard points too far to the right from the center
        for i in range(center_idx, len(coords)):
            if edist(coords[i], coords[center_idx]) > dist_thr:
                labels[i:] = SACCADE
                break

    return gaze_labels


def ivt_group_fixations(gaze_labels, distances, d_treshold=0.5):
    i = 0
    j = 0
    while i < (len(gaze_labels) - 1):
        # Skip saccades to find the first fixation
        if gaze_labels[i] != FIXATION:
            i += 1
            continue

        # Find end of the fixation
        if gaze_labels[i + 1] != SACCADE:
            i += 1
            continue

        # Find the following fixation
        j = i + 1
        while j < len(gaze_labels):
            if gaze_labels[j] == FIXATION:
                break
            if j >= len(gaze_labels) - 1:
                break
            j += 1

        # End because there are no more fixations in the data
        if gaze_labels[j] != FIXATION:
            break

        # Calculate distance between the fixations
        dist = np.sum(distances[i:j])

        # If distance between fixations is short,
        # merge those fixations
        if dist < d_treshold:
            gaze_labels[i:j] = FIXATION

        i = j

    return gaze_labels


def ivt(coords, threshold=50, frequency=1000):
    time = 1 / frequency

    # Calculate distancies between the points
    dists = distances(coords)
    # Calculate velocities
    vels = dists / time
    # Denoising
    vels = savgol_filter(vels, 19, 3)
    # Find fixations and saccades
    gaze_labels = vels < threshold
    # Group fixations that are too close together
    gaze_labels = ivt_group_fixations(gaze_labels, dists)
    # Trim onsets of fixations
    gaze_labels = ivt_trim_fix_onsets(gaze_labels, coords)
    # Filter out short fixations
    gaze_labels = ivt_filter_short_fixations(gaze_labels, frequency)

    return gaze_labels


if __name__ == '__main__':
    # Create a directory for figures
    os.makedirs('fig', exist_ok=True)

    # Load data and convert units to degrees
    data = load_data('train.mat', ['s4', 's14', 's24', 's34', 's10', 's20'])
    data = convert_data_to_deg(data)

    # Initialise CSV generator
    csv = CSVGenerator('output.csv')

    # For each measurement calculate gaze labels
    for sid, measurements in data.items():
        # Calculate Mean Saccade Amplitudes and Mean Fixation Durations
        MSAs_t = msa(measurements)
        MFDs_t = mfd(measurements)
        csv.append_row(sid, MFDs_t, MSAs_t)

        for mid, measurement in enumerate(measurements):
            known = measurement[0]
            coords = measurement[1]
            gaze_labels = ivt(coords)

            # Plot gaze labels
            plot_gaze(coords, gaze_labels, sid, mid)
    csv.close()

    # Plot mfd means and msa bar charts
    msa_mean_known, msa_mean_unknown, msa_std_known, msa_std_unknown = agg_msa(data)
    indexes = (1, 2)
    msa_mean = (msa_mean_known, msa_mean_unknown)
    msa_std = (msa_std_known, msa_std_unknown)
    plt.bar(indexes, msa_mean, yerr=msa_std)
    plt.ylabel(r'Amplitude [$^{\circ}$]')
    plt.xticks(indexes, ('known', 'unknown'))
    plt.savefig('fig/MSA_agg.pdf')
    plt.show()

    # Plot mfd means and msa bar charts
    mfd_mean_known, mfd_mean_unknown, mfd_std_known, mfd_std_unknown = agg_mfd(data)
    indexes = (1, 2)
    mfd_mean = (mfd_mean_known, mfd_mean_unknown)
    mfd_std = (mfd_std_known, mfd_std_unknown)
    plt.bar(indexes, mfd_mean, yerr=mfd_std)
    plt.xticks(indexes, ('known', 'unknown'))
    plt.ylabel('Duration [s]')
    plt.savefig('fig/MFD_agg.pdf')
    plt.show()



