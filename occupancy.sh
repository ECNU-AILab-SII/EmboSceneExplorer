#!/bin/bash

xvfb-run -a -s "-screen 0 1920x1080x24" python src/occupancy/habitatocc.py --data_path example_data/demo
