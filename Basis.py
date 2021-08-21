import numpy as np 

Basis = [
    {
        'name': 'Square',
        'unit_cell': [[0,0]],
        'vector': [[1,0],[0,1]],
    },
    {
        'name': 'Triangle_YC',
        'unit_cell': [[0,0]],
        'vector': [[np.sqrt(3)/2, 1/2],[0,1]],
    },
    {
        'name': 'Honeycomb_YC',
        'unit_cell': [[0,0], [0,np.sqrt(3)/3]],
        'vector': [[1,0], [1/2,np.sqrt(3)/2]],    
    },
    {
        'name': 'Kagome_YC',
        'unit_cell': [[0,0], [-np.sqrt(3)/4,1/4], [0,1/2]],
        'vector': [[np.sqrt(3)/2, -1/2], [0,1]],
    },
]