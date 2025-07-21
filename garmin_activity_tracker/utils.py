"""
Filename: utils.py
Description: Extra functions for Garmin Activity Tracker - ensuring directories exists
Author: Ryan Kari
License: MIT
Created: 2025-07-20
"""
import os

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)