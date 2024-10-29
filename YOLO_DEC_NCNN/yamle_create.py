import yaml

data = {
    "train" : '/school_dataset/train',
        "val" : '/school_dataset/val',
        "test" : '/school_dataset/test',
        "names" : {0 : 'car', 1 : 'person'}}

with open('./school_dataset.yaml', 'w') as f:
    yaml.dump(data, f)

#check written file
with open('./school_dataset.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)