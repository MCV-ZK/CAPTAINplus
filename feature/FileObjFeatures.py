import tqdm
import os

def get_path_vocb(path_set):
    path_vocb = []
    for path in tqdm.tqdm(path_set):
        if path.startswith('/'):
            path_tree = path[1:].split('/')
        else:
            path_tree = path.split('/')
        path_vocb.extend(path_tree)

    path_vocb = dict(Counter(path_vocb))
    path_vocb = sorted(path_vocb.items(),key=lambda x:x[1],reverse=True)
    
    return path_vocb

def get_one_hot_encoding(path_tree, path_vocb):
    oh_vector = []
    for dir in path_tree:
        if dir in path_vocb:
            oh_vector.append(path_vocb[dir])
    
    return oh_vector

def load_path_vocb(vocb_file):
    with open(vocb_file,'r') as fin:
        data = fin.readlines()
    data = data[:10000]
    data = [(item[:-1].split(',')[0],int(item[:-1].split(',')[1])) for item in data]
    
    return data


ExtensionNameType = [
    'pdf',
    'doc',
    'docx',
    'xml',
    'xlsx',
    'cpp'
]

extentsion_name_type = {}
for i, item in enumerate(ExtensionNameType):
    extentsion_name_type[item] = i+1

DirNameType = set([
    'usr','sys','run','sbin','etc',
    'var','home','maildrop','stat',
    'active','incoming','tmp','media',
    'root','data','dev','proc','lib64','lib','bin'
])

dir_name_type = {}

for i, item in enumerate(list(DirNameType)):
    dir_name_type[item] = i+1


from collections import Counter
import json
import pandas as pd
import torch


def main():
    feature_path = './results/features/E31-trace/FileObject.json'
    vector_dir = './results/features/E31-trace/feature_vectors/'
    with open(feature_path,'r') as fin:
        node_features = json.load(fin)

    df = pd.DataFrame.from_dict(node_features,orient='index')

    path_list = df['path'].unique()
    node_type = 'FileObject'

    # path_vocb_freq = get_path_vocb(path_list)
    # with open('results/path_vocabulary.csv','w') as fout:
    #     for item in path_vocb:
    #         fout.write('{},{}\n'.format(item[0],item[1]))
    # path_vocb = path_vocb[:10000]

    path_vocb_freq = load_path_vocb('./results/path_vocabulary.csv')

    path_vocb = {}
    for i, item in enumerate(path_vocb_freq):
        path_vocb[item[0]] = i+1

    path_feature_map = {}
    for path in path_list:
        r_dir = ''
        ext = ''
        if path.startswith('/'):
            path_tree = path[1:].split('/')
        else:
            path_tree = path.split('/')
        r_dir = path_tree[0]
        f_name = path_tree[-1].split('.')
        if len(f_name) >= 2:
            ext = f_name[-1]
        
        # path_feature_map[path] = [dir_name_type.get(r_dir,0),extentsion_name_type.get(ext,0)]
        path_feature_map[path] = [get_one_hot_encoding(path_tree, path_vocb),extentsion_name_type.get(ext,0)]

    df['features'] = df['path'].map(path_feature_map)
    old_features = df['features'].to_list()
    subtypes = df['FileObjectType'].to_list()
    new_features = []
    for i in range(len(old_features)):
        new_features.append([old_features[i][0],old_features[i][1],subtypes[i]])

    df['features'] = new_features
    feature_df = df.drop(columns=['path', 'FileObjectType'])
    feature_df.to_json(os.path.join(vector_dir,'{}.json'.format(node_type)), orient='index')

if __name__ == "__main__":
    main()
