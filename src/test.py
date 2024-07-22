from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pm4py.algo.conformance.alignments.petri_net.algorithm import apply_trace, Variants
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
import os
import sys
import time

def compute_alignment(trace, net, im, fm):
    alignment = apply_trace(trace, net, im, fm)
    return alignment

log_path = "../log/road_variants.xes"
model_path = '../model/road_fodina.pnml'

# Load event log
event_log = xes_importer.apply(log_path)

# Load Petri net
net, im, fm = pnml_importer.apply(model_path)

# Convert event log to a list of traces
traces = list(event_log)


for idx in range(0, len(event_log)):
    result = compute_alignment(traces[idx], net, im, fm)
    print("result: ", result)