# +
import numpy as np

PADDING = 0

# 10x10 square represents one mPMT
# List of top-left pixel positions (row,col) for 2x2 grids representing PMTs 0 to 18
POS_MAP = [(8,4), #0
           (7,2), #1
           (6,0), #2
           (4,0), #3
           (2,0), #4
           (1,1), #5
           (0,4), #6
           (1,6), #7
           (2,8), #8
           (4,8), #9
           (6,8), #10
           (7,6), #11
           # Inner ring
           (6,4), #12
           (5,2), #13
           (3,2), #14
           (2,4), #15
           (3,6), #16
           (5,6), #17
           (4,4)] #18


# -

def get_plot_array(event_data):
    
    # Assertions on the shape of the data and the number of input channels
    assert(len(event_data.shape) == 3 and event_data.shape[2] == 19)
    
    # Extract the number of rows and columns from the event data
    rows = event_data.shape[0]
    cols = event_data.shape[1]
    
    # Make empty output pixel grid
    output = np.zeros(((10+PADDING)*rows, (10+PADDING)*cols))
    
    i, j = 0, 0
    
    for row in range(rows):
        j = 0
        for col in range(cols):
            pmts = event_data[row, col]
            tile(output, (i, j), pmts)
            j += 10 + PADDING
        i += 10 + PADDING
        
    return output


def tile(canvas, ul, pmts):
    
    # First, create 10x10 grid representing single mpmt
    mpmt = np.zeros((10, 10))
    for i, val in enumerate(pmts):
        mpmt[POS_MAP[i][0]][POS_MAP[i][1]] = val

    # Then, place grid on appropriate position on canvas
    for row in range(10):
        for col in range(10):
            canvas[row+ul[0]][col+ul[1]] = mpmt[row][col]


# +
def GenMapping(csv_file):
    mPMT_to_index = {}
    with open(csv_file) as f:
        rows = f.readline().split(",")[1:]
        rows = [int(r.strip()) for r in rows]

        for line in f:
            line_split = line.split(",")
            col = int(line_split[0].strip())
            for row, value in zip(rows, line_split[1:]):
                value = value.strip()
                if value: # If the value is not empty
                    mPMT_to_index[int(value)] = (row, col)
    return mPMT_to_index

#mPMT_to_index = GenMapping(csv_file)
#PMT_to_index = {k*19+18:v for k,v in mPMT_to_index.items()}
#xy_to_PMT = {v:k for k,v in PMT_to_index.items()}


# +
def gen_mpmt_mapping():
    xs = []
    ys = []
    mapping = []
    # Outer ring
    for i in range(12):
        theta = 2*np.pi*i/12
        mapping.append([-0.4*np.sin(theta), -0.4*np.cos(theta)])
    # Inner ring
    for i in range(6):
        theta = 2*np.pi*i/6
        mapping.append([-0.2*np.sin(theta), -0.2*np.cos(theta)])
    # Center point
    mapping.append([0, 0])
    return mapping

def mpmt_mapping(_pmt_index):
    _pmt_in_module_id = _pmt_index%19
    _module_index = _pmt_index//19
    if _pmt_in_module_id >= 0 and _pmt_in_module_id<12:
        _theta = 2*np.pi*_pmt_in_module_id/12
        _radius = 0.4
    elif _pmt_in_module_id >= 12 and _pmt_in_module_id<18:
        _theta = 2*np.pi*(_pmt_in_module_id-12)/6
        _radius = 0.2
    else:
        _theta = 0
        _radius = 0
    if not is_barrel(_module_index):
        _theta = _theta+np.pi
    _mapping = [-_radius*np.sin(_theta), -_radius*np.cos(_theta)]
    return _mapping
    
#PMT_to_index = gen_mpmt_mapping()


# +
def is_barrel(module_index):
    """Returns True if module is in the Barrel"""
    return ( (module_index<600) | ((module_index>=696)&(module_index<736)) )

def is_bottom(module_index):
    """Returns True if module is in the bottom cap"""
    return ( (module_index>=600)&(module_index<696) )

def is_top(module_index):
    """Returns True if module is in the top cap"""
    return ( (module_index>=736)&(module_index<832) )
# -


