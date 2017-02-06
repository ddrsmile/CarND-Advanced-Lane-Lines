# -*- coding: utf-8 -*-
import os

ROOT = os.path.abspath(os.path.dirname(os.pardir))
CALI_IMG = 'camera_cal'
TEST_IMG = 'test_images'
OUTPUT_IMG = 'output_images'

cali_path = os.path.join(ROOT, CALI_IMG)
test_path = os.path.join(ROOT, TEST_IMG)
out_path = os.path.join(ROOT, OUTPUT_IMG)
