import json
import os
dir = os.path.dirname(__file__)
file_path = os.path.join(
    dir, 'input-sample-multiple.txt')

saved_file_path = os.path.join(
    dir, 'input-sample.txt'
)

with open(file_path, 'r') as json_file:
    output = json.load(json_file)

output['output']['person'] = [output['output']['person'][0]]

with open(saved_file_path, 'w') as json_write:
    json.dump(output, json_write)
