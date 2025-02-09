import logging
import os
import argparse
import json
import time
import pickle
import pandas as pd
import numpy as np
# # Overhead
# import psutil
# import csv
# current_process = psutil.Process(os.getpid())
# perf_file = open('system_usage.csv', 'a', newline='')
# writer = csv.writer(perf_file)
# writer.writerow(['Time', 'Memory Usage (MB)'])
from graph.Subject import Subject
import resource
from datetime import datetime
import pytz
from parse.nodeParsing import parse_subject as parse_subject_
from pympler import asizeof

from datetime import datetime
from utils.utils import *
from utils.eventClassifier import eventClassifier
from model.captain import CAPTAIN
from graph.Event import Event
from utils.graph_detection import add_nodes_to_graph
from pathlib import Path
from collections import Counter
import pdb
import csv
def start_experiment(args):
    experiment = Experiment(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), args, args.experiment_prefix)
    # start_user_cpu_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    # start_system_cpu_time = resource.getrusage(resource.RUSAGE_SELF).ru_stime
    mo = CAPTAIN(att = args.att, decay = args.decay)
    mo.mode = 'eval'
    # mo.mode = 'train'
    eventcount = 0
  
    print("Begin preparing testing...")
    logging.basicConfig(level=logging.INFO,
                        filename='debug.log',
                        filemode='w+',
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    experiment.save_hyperparameters()
    ec = eventClassifier(args.ground_truth_file)
        
    if args.param_path:
        with open(os.path.join(args.param_path, 'train', 'params/lambda-e{}.pickle'.format(args.model_index)), 'rb') as fin:
            mo.lambda_dict = pickle.load(fin)
        with open(os.path.join(args.param_path, 'train', 'params/tau-e{}.pickle'.format(args.model_index)), 'rb') as fin:
            mo.tau_dict = pickle.load(fin)
        with open(os.path.join(args.param_path, 'train', 'params/alpha-e{}.pickle'.format(args.model_index)), 'rb') as fin:
            mo.alpha_dict = pickle.load(fin)
                
    # close interval
    if args.time_range:
        detection_start_time = args.time_range[0]
        detection_end_time = args.time_range[1]
    else:
        detection_start_time = 0
        detection_end_time = 1e21

    Path(os.path.join(experiment.get_experiment_output_path(), 'alarms')).mkdir(parents=True, exist_ok=True)
    mo.alarm_file = open(os.path.join(experiment.get_experiment_output_path(), 'alarms/alarms-in-test.txt'), 'a')

    log_file = os.path.join(args.data_path, 'logs.json')
    node_buffer = {}
    loaded_line = 0
    fp_counter = {}
    # false_alarms = []
    experiment.alarm_dis = Counter([])
    total_loss = 0
    ## alarm node evaluation
    alarm_nodes = set()
    count1=0
    fpcount=0
    tpnumber=0
    fpnumber=0
    experiment.detection_time = 0

    decoder = json.JSONDecoder()
    with open(log_file, 'r') as fin:
        for line in fin:
            detection_delay_marker = time.time()
            loaded_line += 1
            if loaded_line == 1:
                begin_time = time.time()
                ## Calculate CPU Time
                start_cpu_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime
            if loaded_line > 0 and loaded_line % 100000 == 0:
                print("CAPTAIN has detected {:,} logs.".format(loaded_line))
                
                dt = datetime.fromtimestamp(prt_ts / 1e9)
                ny_tz = pytz.timezone('America/New_York')
                ny_dt = dt.astimezone(ny_tz)
                ny_dt_str = ny_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                print(ny_dt_str)
                
                delta_time = time.time() - begin_time
                begin_time = time.time()
                print(f"Detection time for 100K logs is {delta_time:.2f} s")
                # # Overhead
                # current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                # # cpu_usage = current_process.cpu_percent()
                # memory_usage = current_process.memory_info()
                # with open('your_log_file.csv', 'a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([current_time, memory_usage.rss/(1024 * 1024)])
                # # print(f"{current_time}, Memory: {memory_usage.rss/(1024 * 1024)}MB")
            log_data = decoder.decode(line)
            if log_data['logType'] == 'EVENT':
                event = Event(None, None)
                event.load_from_dict(log_data['logData'])
                if event.type == 'UPDATE':
                    if 'exec' in event.value:
                        if event.nid in mo.Nodes:
                            mo.Nodes[event.nid].processName = event.value['exec']
                        elif event.nid in node_buffer:
                            node_buffer[event.nid]['processName'] = event.value['exec']
                    elif 'name' in event.value:
                        if event.nid in mo.Nodes:
                            mo.Nodes[event.nid].name = event.value['name']
                            mo.Nodes[event.nid].path = event.value['name']
                        elif event.nid in node_buffer:
                            node_buffer[event.nid]['name'] = event.value['name']
                            node_buffer[event.nid]['path'] = event.value['name']
                    elif 'cmdl' in event.value:
                        if event.nid in mo.Nodes:
                            mo.Nodes[event.nid].cmdLine = event.value['cmdl']
                        elif event.nid in node_buffer:
                            node_buffer[event.nid]['cmdLine'] = event.value['cmdl']
                # elif event.type == 'OBJECT_VERSION_UPDATE':
                #     if event.old in mo.Nodes and event.new in node_buffer:
                #         add_nodes_to_graph(mo, event.new, node_buffer[event.new])
                #         del node_buffer[event.new]
                #         mo.Nodes[event.new].setObjTags(mo.Nodes[event.old].tags()[2:])
                #         if mo.mode == 'train':
                #             mo.Nodes[event.new].set_grad(mo.Nodes[event.old].get_grad())
                #             mo.Nodes[event.new].set_lambda_grad(mo.Nodes[event.old].get_lambda_grad())
                #         # del mo.Nodes[event.old]
                else:
                    prt_ts = event.time
                    if event.time < detection_start_time:
                        continue
                    elif event.time > detection_end_time:
                        break
      
                    if event.src not in mo.Nodes and event.src in node_buffer:
                        add_nodes_to_graph(mo, event.src, node_buffer[event.src])
                        del node_buffer[event.src]

                    if isinstance(event.dest, str) and event.dest not in mo.Nodes and event.dest in node_buffer:
                        add_nodes_to_graph(mo, event.dest, node_buffer[event.dest])
                        del node_buffer[event.dest]

                    if isinstance(event.dest2, str) and event.dest2 not in mo.Nodes and event.dest2 in node_buffer:
                        add_nodes_to_graph(mo, event.dest2, node_buffer[event.dest2])
                        del node_buffer[event.dest2]

                    gt = ec.classify(event.id)
                    diagnosis = mo.add_event(event,gt)
                    experiment.update_metrics(diagnosis, gt)

                    if(diagnosis != None and gt != diagnosis ) or (diagnosis == None and gt != None):
                        s_labels, o_labels, pc, lambda_grad, thr_grad, loss = mo.add_event_online_loss(event,gt)                       
                    #     pdb.set_trace()
                    # add incremental learning here:
                        total_loss += loss  
                        #A subject   
                        if diagnosis != None and gt != diagnosis:                          
                            fpcount+=1
                            flag=0
                            args.lr=0.001*(fpcount/50+1)                       
                        if diagnosis == None and gt != None:
                            args.lr=0.25
                            fpcount=fpcount/2
                            flag=1                                
                        node_gradients = []
                        if s_labels:      #set different args.lr and update rangeability, flag=1 means fn
                            node_gradients.extend(s_labels)   # update s_labels corresponding mo.alpha_dict[key]
                            intg_nid_labels = {}
                            conf_nid_labels = {}   
                            for item in node_gradients:
                                if item[0][1] == 'i':
                                    if item[0] not in intg_nid_labels:
                                        intg_nid_labels[item[0]] = []
                                    intg_nid_labels[item[0]].append(item[1])
                                elif item[0][1] == 'c':
                                    if item[0] not in conf_nid_labels:
                                        conf_nid_labels[item[0]] = []
                                    conf_nid_labels[item[0]].append(item[1])     
                            node_iTagGradients_dict = {}
                            for node, value in intg_nid_labels.items():
                                node_name = mo.Nodes[node[0]].get_name()
                                node_type = mo.Nodes[node[0]].type
                                if (node_type, node_name) not in node_iTagGradients_dict:
                                    node_iTagGradients_dict[(node_type, node_name)] = []
                                node_iTagGradients_dict[(node_type, node_name)].extend(value)  
                            for key in node_iTagGradients_dict.keys():
                                node_iTagGradients_dict[key] = sum(node_iTagGradients_dict[key])     
                            for key, item in node_iTagGradients_dict.items():
                                if key not in mo.alpha_dict:
                                    mo.alpha_dict[key] = mo.get_default_a(key[0], key[1])[0] 
                                mo.alpha_dict[key] -= args.lr*item
                                mo.alpha_dict[key] = np.clip(mo.alpha_dict[key], 0.0, 1.0)
                                default_a0 = mo.get_default_a(key[0], key[1])[0]                                                            
                                if np.absolute(mo.alpha_dict[key] - default_a0) <= 1e-6:    #delete value too small
                                    del mo.alpha_dict[key] 
                                #if key in mo.alpha_dict:
                                mo.update_stags(event.src,event,gt,flag,fpcount)   #update node subject tag                                                                                                                                                                
                        # A object:
                        node_gradients = []                       
                        if o_labels:
                            node_gradients.extend(o_labels)  # update s_labels corresponding mo.alpha_dict[key]
                            intg_nid_labels = {}
                            conf_nid_labels = {}
                            for item in node_gradients:
                                if item[0][1] == 'i':
                                    if item[0] not in intg_nid_labels:
                                        intg_nid_labels[item[0]] = []
                                    intg_nid_labels[item[0]].append(item[1])
                                elif item[0][1] == 'c':
                                    if item[0] not in conf_nid_labels:
                                        conf_nid_labels[item[0]] = []
                                    conf_nid_labels[item[0]].append(item[1])
                                
                            node_iTagGradients_dict = {}
                            for node, value in intg_nid_labels.items():
                                node_name = mo.Nodes[node[0]].get_name()
                                node_type = mo.Nodes[node[0]].type
                                if (node_type, node_name) not in node_iTagGradients_dict:
                                    node_iTagGradients_dict[(node_type, node_name)] = []
                                node_iTagGradients_dict[(node_type, node_name)].extend(value)
                            for key in node_iTagGradients_dict.keys():
                                node_iTagGradients_dict[key] = sum(node_iTagGradients_dict[key])
                            for key, item in node_iTagGradients_dict.items():
                                if key not in mo.alpha_dict:
                                    mo.alpha_dict[key] = mo.get_default_a(key[0], key[1])[0]
                                mo.alpha_dict[key] -= args.lr*item
                                mo.alpha_dict[key] = np.clip(mo.alpha_dict[key], 0.0, 1.0)
                                default_a0 = mo.get_default_a(key[0], key[1])[0]      
                                if np.absolute(mo.alpha_dict[key] - default_a0) <= 1e-6:
                                    del mo.alpha_dict[key]           #delete value too small
                                mo.update_otags(event.dest,event,gt,flag,fpcount) #update node object tag 
                        #mo.updateg(event,gt)  

                        if diagnosis != None and gt != diagnosis: 
                            args.lr=0.01*(fpcount/50+1)            
                        #args.lr=0.001
                        if diagnosis == None and gt != None:
                            args.lr=0.05                   
                        # G update
                        edge_gradients = []                       
                        edge_gradients.extend(lambda_grad) 
                        if len(edge_gradients) > 0:
                            seo_lambda_gradients = {}
                            for item in edge_gradients:
                                if item[0] not in seo_lambda_gradients:
                                    seo_lambda_gradients[item[0]] = 0
                                seo_lambda_gradients[item[0]] += item[1]
                            for key, value in seo_lambda_gradients.items():
                                if key not in mo.lambda_dict:
                                    mo.lambda_dict[key] = 0
                                # mo.lambda_dict[key] = mo.lambda_dict[key] - args.lr * value/experiment.fp
                                mo.lambda_dict[key] = mo.lambda_dict[key] - args.lr*value                                
                                mo.lambda_dict[key] = np.clip(mo.lambda_dict[key], 0.0, 1.0)
                                total_loss += args.gamma * mo.lambda_dict[key] * mo.lambda_dict[key]
                                if mo.lambda_dict[key] <= 1e-6:
                                    del mo.lambda_dict[key]   

                        # T update
                        args.lr=0.001
                        if diagnosis == None and gt != None:
                            args.lr=0.001
                        if diagnosis == None and gt != None:
                            args.lr=0.01
                        for key, value in thr_grad.items():
                            if key not in fp_counter:
                                fp_counter[key] = [0, 0, 0, 0, 0, 0, 0, 0]
                            for i, grad in enumerate(value):
                                if grad:
                                    fp_counter[key][i] = grad
                            if key not in mo.tau_dict.keys():
                                mo.tau_dict[key] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                               
                            for i, value in enumerate(fp_counter[key]):
                                mo.tau_dict[key][i] -= args.lr*value
                                mo.tau_dict[key][i] = max(0, mo.tau_dict[key][i])
                            for i in range(len(mo.tau_dict[key])):
                                mo.tau_dict[key][i] = min((mo.tau_dict[key][i] + args.lr * args.tau * (0.5-mo.tau_dict[key][i])), 0.5)
                            total_loss += args.tau * np.linalg.norm(np.array(mo.tau_dict[key]) - 0.5)
                            if np.all(np.isclose(mo.tau_dict[key], 0.5)):
                                del mo.tau_dict[key] 

                    if diagnosis != None:
                        if diagnosis == 'FileCorruption':
                            alarm_nodes.add(event.src)
                        else:
                            alarm_nodes.add(event.src)
                            alarm_nodes.add(event.dest)
                            alarm_nodes.add(event.dest2)

                    if gt == None and diagnosis != None:
                        # false_alarms.append(diagnosis)
                        experiment.alarm_dis[diagnosis] += 1
            elif log_data['logType'] == 'NODE':
                node_buffer[log_data['logData']['id']] = log_data['logData']
                del node_buffer[log_data['logData']['id']]['id']
                # print(f'Size of node buffer {len(node_buffer)}')
            elif log_data['logType'] == 'PRINCIPAL':
                mo.Principals[log_data['logData']['uuid']] = log_data['logData']
                del mo.Principals[log_data['logData']['uuid']]['uuid']
            elif log_data['logType'] == 'CTL_EVENT_REBOOT':
                # mo.reset()
                # node_buffer = {}
                # pdb.set_trace()
                pass
            
            experiment.detection_time += time.time()-detection_delay_marker
            
    # # Calculate CPU Time
    # end_cpu_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    # print(f"CPU Time used for detection: {end_cpu_time - start_cpu_time} s")
    
    # # Calculate Max Memory Usage
    # max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print(f"Memory usage for detection: {max_rss} KB")
    
    # experiment.detection_time = time.time()-begin_time
    print('The detection time is :{:.2f} s'.format(experiment.detection_time))
    print('The event throughput is :{:.2f} s'.format(loaded_line/experiment.detection_time))
    print(fpnumber,tpnumber)
    # print("{} Mb".format(asizeof.asizeof(mo)/(1024*1024)))
    print("# of nodes: {}".format(len(mo.Nodes)))

    ## Alarm Nodes
    alarm_nodes = alarm_nodes - {None}
    with open(os.path.join(experiment.get_experiment_output_path(), 'alarms/alarms-nodes.txt'), 'w') as fout:
        for nid in alarm_nodes:
            print(nid, file=fout)
            
    ## For Theia
    # from graph.Object import Object
    # from graph.Subject import Subject
    # node_names = set()
    # with open(os.path.join(experiment.get_experiment_output_path(), 'alarms/alarms-nodes-name.txt'), 'w') as fout:
    #     for nid in alarm_nodes:
    #         if isinstance(mo.Nodes[nid], Subject):
    #             nname = f"{mo.Nodes[nid].pid} {mo.Nodes[nid].get_name()} {mo.Nodes[nid].get_cmdln()}"
    #         elif isinstance(mo.Nodes[nid], Object):
    #             nname = mo.Nodes[nid].get_name()
    #         if nname not in node_names:
    #             print(nname, file=fout)
    #             node_names.add(nname)
              
    mo.alarm_file.close()
    # experiment.alarm_dis = Counter(false_alarms)
  
    #experiment.print_metrics(TP,eventcount,loaded_line)
    experiment.print_metrics()
    experiment.save_metrics()
    ec.analyzeFile(open(os.path.join(experiment.get_experiment_output_path(), 'alarms/alarms-in-test.txt'),'r'))
    ec.summary(os.path.join(experiment.metric_path, "ec_summary_test.txt"))
    print("Metrics saved in {}".format(experiment.get_experiment_output_path()))

    ## Overhead
    # perf_file.close()
    
    
def main():
    parser = argparse.ArgumentParser(description="train or test the model")
    parser.add_argument("--att", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--ground_truth_file", type=str)
    parser.add_argument("--data_path", nargs='?', type=str)
    parser.add_argument("--param_type", type=str)
    parser.add_argument("--model_index", type=int)
    parser.add_argument("--experiment_prefix", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--param_path", type=str)
    parser.add_argument("--time_range", nargs=2, type=str, default = None)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--tau", type=float, default=0)

    args = parser.parse_args()
    if args.time_range:
        args.time_range[0] = (datetime.timestamp(datetime.strptime(args.time_range[0], '%Y-%m-%dT%H:%M:%S%z')))*1e9
        args.time_range[1] = (datetime.timestamp(datetime.strptime(args.time_range[1], '%Y-%m-%dT%H:%M:%S%z')))*1e9

    start_experiment(args)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("main()")
    main()