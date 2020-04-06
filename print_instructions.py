import re
import os


def print_instructions():
    usr=os.environ['USER']
    with open('ip.txt') as f:
        myip = f.readline()
    myip=myip.rstrip()
    
    textfile = open('jupyter_logbook.txt', 'r')
    matches = []
    reg = re.compile('^\s* or http://127.0.0.1:([0-9]*)')
    for line in textfile:
        match=reg.match(line)
        if match is not None:
            port=match.group(1)
            print("In a terminal running local shell on your laptop paste:")
            print('ssh -L {}:localhost:{} {}@{} -N -f'.format(port, port, usr, myip))
            print("then in your browser address bar paste:")
            print(line.lstrip())
            
    textfile.close()




if __name__=='__main__':
    print_instructions()
