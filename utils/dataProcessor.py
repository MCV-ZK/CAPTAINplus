import json
import configparser
import os
import glob

class dataProcessor:
    def __init__(self, configFile):
        self.config = configparser.ConfigParser()
        self.config.read(configFile)
        self.dpConfig = self.config['DataProcessor']
        self.fromFormat = self.dpConfig['FromFileFormat']
        self.toDir = self.dpConfig['ToDir']
        self.vertexDataFileName = self.dpConfig['VertexDataFileName']
        self.edgeDataFileName = self.dpConfig['EdgeDataFileName']
        self.segSize = int(self.dpConfig['SegmentSize'])
        self.startTimestamp = int(self.dpConfig['StartTimestamp'])
        self.endTimestamp = int(self.dpConfig['EndTimestamp'])
        self.unusedEventType = ['EVENT_ACCEPT', 'EVENT_ADD_OBJECT_ATTRIBUTE', 'EVENT_BIND', 'EVENT_BLIND', \
                                'EVENT_BOOT', 'EVENT_CHECK_FILE_ATTRIBUTES', 'EVENT_FCNTL', 'EVENT_LOGCLEAR', \
                                'EVENT_LOGIN', 'EVENT_LOGOUT', 'EVENT_LSEEK', 'EVENT_TRUNCATE', 'EVENT_UNIT', \
                                'EVENT_UNLINK', 'EVENT_UPDATE', 'EVENT_WAIT']

    def separate(self):
        if not os.path.exists(self.toDir):
            os.makedirs(self.toDir)
        vertexPath = os.path.join(self.toDir, self.vertexDataFileName)
        totalCounter = 0
        vertexCounter = 0
        edgeCounter = 0
        segCounter = 0
        edgeIndex = 0
        edgePath = os.path.join(self.toDir, self.edgeDataFileName + '.' + str(edgeIndex))
        files = glob.glob(self.fromFormat)
        files.sort(key=lambda f:int(os.path.splitext(f)[-1][1:]))
        e = open(edgePath, 'w+')
        with open(vertexPath, 'w+') as v:
            for file in files:
                with open(file, 'r') as f:
                    for line in f:
                        totalCounter += 1
                        tmp = json.loads(line.strip())
                        line = line.strip()
                        line = line + '\n'
                        if 'com.bbn.tc.schema.avro.cdm18.Subject' in tmp['datum'] or \
                        'com.bbn.tc.schema.avro.cdm18.SrcSinkObject' in tmp['datum'] or \
                        'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in tmp['datum'] or \
                        'com.bbn.tc.schema.avro.cdm18.FileObject' in tmp['datum'] or \
                        'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in tmp['datum'] or \
                        'com.bbn.tc.schema.avro.cdm18.Principal' in tmp['datum'] or \
                        'com.bbn.tc.schema.avro.cdm18.MemoryObject' in tmp['datum']:
                            v.write(line)
                            vertexCounter += 1
                        elif 'com.bbn.tc.schema.avro.cdm20.Event' in tmp['datum']:
                            timestamp = int(tmp['datum']['com.bbn.tc.schema.avro.cdm20.Event']['timestampNanos'])
                            event_type = tmp['datum']['com.bbn.tc.schema.avro.cdm20.Event']['type']
                            if (timestamp < self.startTimestamp) or (timestamp > self.endTimestamp and self.endTimestamp > 0):
                                continue
                            if event_type in self.unusedEventType:
                                continue
                            e.write(line)
                            segCounter += 1
                            edgeCounter += 1
                            if edgeCounter % 1000000 == 0:
                                print(f"{edgeCounter//1000000}M lines of data processed.")
                            if segCounter == self.segSize and self.segSize > 0:
                                e.close()
                                edgeIndex += 1
                                segCounter = 0
                                edgePath = os.path.join(self.toDir, self.edgeDataFileName + '.' + str(edgeIndex))
                                e = open(edgePath, 'w+')
        print("Summary:")
        print(f"read in {totalCounter} lines of data.")
        print(f"output {vertexCounter} lines of vertex data.")
        print(f"output {edgeCounter} lines of edge data.")
        print(f"filtered {totalCounter-vertexCounter-edgeCounter} lines of data.")
