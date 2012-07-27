#! /usr/bin/env python

import sys

def readMesh(fileName, nodes='nodes.dat', elements='elements.dat'):
    fileRead = open(fileName, 'r')
    lines =  fileRead.readlines()
    fileRead.close()
    fileWriteNodes = open(nodes, 'w')
    fileWriteElements = open(elements, 'w')
    num = 1
    for line in lines[3:]:
        if 'Nodes' in line:
            mode = 'n'
        elif 'Elements' in line:
            mode = 'e'
        tokens = line[:-1].split()
        if len(tokens) > 1:
            if mode == 'n':
                fileWriteNodes.write('%s\t%s\t%s\t%s\n' % (tokens[0], tokens[1], tokens[2], tokens[3]))
            elif mode == 'e':
                if tokens[1] == '2':
                    fileWriteElements.write('%s\t%s\t%s\t%s\t%s\n' % (num, tokens[-3], tokens[-2], tokens[-1], 1))
                    num += 1
    fileWriteNodes.close()
    fileWriteElements.close()
    
try:
    inputRaw = sys.argv[1]
except IndexError:
    print "invalid arguments"
    sys.exit()
try:
    argvs = sys.argv[2:]
    kvargs = {}
    for uInput in argvs:
        if 'nodes' in uInput:
            kvargs['nodes'] = uInput.split('=')[1]
        elif 'elements' in uInput:
            kvargs['elements'] = uInput.split('=')[1]
except IndexError:
    kvargs = {}
        


readMesh(inputRaw, **kvargs)
print "All done, boss!"
