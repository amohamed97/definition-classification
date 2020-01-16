from pathlib import Path
import sys
import pandas as pd
import re
import csv


sentences = []
for child in Path("./output_test/").iterdir():
        if child.suffix == '.deft':
            with open(child) as f:
                lines = list(f.readlines())
                for line in lines:
                    splitted = line.split('"')
                    text = splitted[1]
                    has_def = splitted[-2]
                    sentences.append("__label__"+str(has_def)+" "+text.strip()+"\n")
with open("output.test","a+") as f:
    for sentence in sentences:
        f.write(sentence)