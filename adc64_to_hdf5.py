#!/usr/bin/env python
#
# Helper script to convert a single adc64 raw data file format to a representative hdf5 format
#
# requires:
#   python>=3.8
#   numpy
#   h5py
#   tqdm
#
# installation:
#   pip install numpy h5py tqdm
#
# usage:
#   python adc64_to_hdf5.py <number of samples in waveforms> <input file, .data> <output file, .h5>
#
# output hdf5 file has the following datasets:
#   "header": adc64 "Time Header"
#   "event":  adc64 "Event Payload"
#   "device": adc64 "Device Payload"
#   "time":   adc64 "MStream Time Block"
#   "data":   adc64 "MStream Data Block"
#   "ref":    region of "data" relevant for given event
#
# contact:
#   Peter Madigan, pmadigan@berkeley.edu
#
import numpy as np
import h5py
import sys
import tqdm
from collections import defaultdict

from adc64format import ADC64Reader

nsamples = int(sys.argv[-3])
input_file = sys.argv[-2]
output_file = sys.argv[-1]

WRITE_BUFFER = 1024-1

dtypes = ADC64Reader.dtypes.copy()
dtypes['data'] = dtypes['data'](nsamples)
dtypes['ref'] = np.dtype([('start','i4'), ('stop','i4')])

print(f'Opening input file {input_file}...', end=' ')
with ADC64Reader(input_file) as reader:
    print(f'Creating new file at {output_file}...', end=' ')
    with h5py.File(output_file, 'w') as fo:

        for key in dtypes:
            fo.create_dataset(key, (0,), maxshape=(None,), dtype=dtypes[key], compression='gzip', shuffle=True)
        dset_keys = list(fo.keys())

        ptr = defaultdict(int)
        data = defaultdict(list)
        print('datasets: ' + ', '.join(dset_keys) + '...', end=' ')

        print('Done!')

        with tqdm.tqdm(unit=' events', smoothing=0) as pbar:
            while True:
                event = reader.next()

                if event is not None and event['header'][0] is not None:
                    for key in event:
                        data[key].append(event[key][0])
                        ptr[key] += len(event[key][0])

                    # create index for matching waveforms to triggers
                    data['ref'].append(np.zeros((1,), dtype=dtypes['ref']))
                    data['ref'][-1]['stop'] = ptr['data']
                    data['ref'][-1]['start'] = ptr['data'] - len(data['data'][-1])
                    ptr['ref'] += 1

                # extend hdf5 file as needed
                for key in dset_keys:
                    if len(fo[key]) <= ptr[key]:
                        fo[key].resize((2*ptr[key],))

                # write data
                if any([(len(data[key]) > WRITE_BUFFER) for key in dset_keys]) or event is None:
                    for key in dset_keys:
                        if (len(data[key]) > 0):
                            write_ptr = ptr[key]
                            write_data = np.concatenate(data[key], axis=0).ravel()

                            #print(key, 'write', write_data.shape, 'to', write_ptr)
                            fo[key].write_direct(write_data, dest_sel=np.s_[write_ptr-len(write_data):write_ptr])
                            
                            data[key] = list()

                if event is None:
                    break
                
                pbar.update()

        for key in dset_keys:
            fo[key].resize((ptr[key],))
            print(f'"{key}" final shape of {ptr[key]}')

print('Conversion complete!')

