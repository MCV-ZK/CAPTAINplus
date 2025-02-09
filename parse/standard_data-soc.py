import json
import os
import argparse
import time
import pdb
import sys
sys.path.extend(['.','..','...'])
from utils.utils import *
from graph.Object import Object
from graph.Event import Event
from graph.Subject import Subject
from datetime import datetime
from dateutil.parser import isoparse

def parse_object_optc(record_datum):
    object_type = record_datum['log_name']
    if object_type == 'PROCESS':
        object = Subject(id=record_datum['uuid'], type = object_type)
    else:
        object = Object(id=record_datum['uuid'], type = object_type)    
    if object_type == 'Network':
        object.type = 'NetFlowObject'
        object.subtype = 'FLOW_SOC'
        if record_datum.get('arguments', {}).get('direction', None) == '1':
            object.set_IP(record_datum['arguments']['source_ip'], record_datum['arguments']['source_port'], None)
        elif record_datum.get('arguments', {}).get('direction', None) == '2':
            object.set_IP(record_datum['arguments']['destination_ip'], record_datum['arguments']['destination_port'], None)
        else:
            return None
    if object_type == 'DNSQuery':
        object = Object(id=record_datum['uuid'], type = object_type)
       
    elif object_type == 'PROCESS':
        return None
    elif object_type == 'Privilege':
        return None

    elif object_type == 'FileMonitor':
        object.type = 'FileObject'
        object.subtype = 'FILE'
        path = record_datum.get('arguments',{}).get('filepath', None)
        if path:
            object.name = path
            object.path = path
        else:
            return None
    elif object_type == 'Tunnel':
        return None
    elif object_type == 'CommandLineAudit':
        return None
    elif object_type == 'ServiceProcess':
        return None
    elif object_type == 'ClientPolicyUpdate':    
        return None
    elif object_type == 'MBR':    
        return None
    return object

def parse_subject_optc(record_datum):
    pid_ = record_datum.get('arguments', {}).get('process_id')
    ppid_ = record_datum.get('arguments', {}).get('parent_process_id')
    principal_ = record_datum.get('user') or record_datum.get('arguments', {}).get('user')
    #tid_ = record_datum['tid']
    arguments = record_datum['arguments']
    process_path = record_datum['arguments'].get('process_path', None)
    if process_path:
        pname_ = process_path.split('\\')[-1]
    else:
        pname_ = None
    parent_ = None
    cmdLine_ = None
    subject = Subject(id=record_datum.get('arguments', {}).get('process_uuid'), type = 'SUBJECT_PROCESS', pid = record_datum.get('arguments', {}).get('process_id'), ppid = ppid_, parentNode = parent_, cmdLine = cmdLine_, processName=pname_)
    return subject
def parse_event_optc(log_data, node_buffer):
    event = Event(log_data['id'], log_data['timestamp'])
    event_type = log_data['operating']
    if log_data['arguments'].get('process_uuid') and log_data['arguments']['process_uuid'] != '00000000-0000-0000-0000-000000000000':
        event.src = log_data['arguments']['process_uuid']
    if log_data['uuid'] and log_data['uuid'] != '00000000-0000-0000-0000-000000000000':
        event.dest = log_data['uuid']
    try:
        if log_data['log_name'] == 'Network':
            if event_type =='Send':
                assert node_buffer.get(event.src, None) and node_buffer.get(event.dest, None)
                event.type = 'write'
            if event_type =='Recv':
                assert node_buffer.get(event.src, None) and node_buffer.get(event.dest, None)
                event.type = 'read'           
        elif log_data['log_name'] == 'FileMonitor':
            assert node_buffer.get(event.src, None) and node_buffer.get(event.dest, None)
            if event_type =='Change':
                event.type = 'write'
            elif event_type =='Read':
                event.type = 'read'
            elif event_type =='Created':
                event.type = 'create'
            elif event_type == 'Deleted':
                event.type = 'remove'
            elif event_type =='Rename':
                event.type = 'rename'
                event.parameters = log_data['arguments']['new_filepath']
            else:
                return None
        elif log_data['log_name'] == 'Process':
            assert node_buffer.get(event.src, None) and node_buffer.get(event.dest, None)
            if event_type =='Created':
                event.type = 'clone'
            elif event_type =='OPEN':
                event.type = 'open_process'
            elif event_type =='Terminated':
                event.type = 'terminate'
            elif event_type =='Update':
                event.type = 'update'
            else:
                return None
        elif log_data['log_name'] == 'DNSQuery':
            if event_type == 'Created':
                event.type = 'chmod'

    except AssertionError as ae:
        return None
    
    if event.type:
        return event
    else:
        return None    
import os
import time
import json
import argparse

def start_experiment(args):
    output_file = open(os.path.join(args.output_data, 'logs.json'), 'w')

    last_event_str = ''
    node_buffer = {}

    ##### Set Up Counters #####
    loaded_line = 0    
    envt_num = 0
    edge_num = 0
    node_num = 0

    begin_time = time.time()
    decoder = json.JSONDecoder()

    # Update volume_list to load .log files from the new directory
    volume_list = []
    log_directory = '/home/yangyangwei/CAPTAIN/CAPTAIN/SOC/L11'
    
    # Get all .log files from the directory
    for filename in os.listdir(log_directory):
        if filename.endswith('.log'):
            volume_list.append(os.path.join(log_directory, filename))

    for volume in volume_list: 
        print("Loading {} ...".format(volume))
        with open(volume, 'r') as fin:
            for line in fin:
                loaded_line += 1
                if loaded_line % 100000 == 0:
                    print("CAPTAIN has parsed {:,} lines.".format(loaded_line))

                line = line.strip()  # 去掉空白字符
                if not line:  # 跳过空行
                    continue

                try:
                    record_datum = decoder.decode(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}, Error: {e}")
                    continue  # 跳过无法解析的行

                record_type = record_datum['operating']
                subject = parse_subject_optc(record_datum)
                if subject and subject.id not in node_buffer:
                    node_buffer[subject.id] = subject
                    log_datum = {'logType':'NODE', 'logData': json.loads(subject.dumps())}
                    output_file.write(json.dumps(log_datum)+'\n')
                    node_num += 1
                
                object = parse_object_optc(record_datum)
                if object and object.id not in node_buffer:
                    node_buffer[object.id] = object
                    log_datum = {'logType':'NODE', 'logData': json.loads(object.dumps())}
                    output_file.write(json.dumps(log_datum)+'\n')
                    node_num += 1

                event = parse_event_optc(record_datum, node_buffer)
                if event:
                    event_str = '{},{},{}'.format(event.src, event.type, event.dest)
                    if event_str != last_event_str:
                        last_event_str = event_str
                        edge_num += 1
                        log_datum = {'logType':'EVENT', 'logData': json.loads(event.dumps())}
                        output_file.write(json.dumps(log_datum)+'\n')

    output_file.close()
    print("Parsing Time: {:.2f}s".format(time.time()-begin_time))
    print("#Events: {:,}".format(loaded_line))
    print("#Nodes: {:,}".format(node_num))
    print("#Edges: {:,}".format(edge_num))
    
    
def main():
    parser = argparse.ArgumentParser(description="Data Standardize")
    parser.add_argument("--input_data", type=str, default = 'data/soc')
    parser.add_argument("--output_data", type=str, default = 'data/soc')
    args = parser.parse_args()

    start_experiment(args)


if __name__ == '__main__':
    main()

