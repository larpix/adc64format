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
#   python adc64_to_hdf5.py <input file, .data> <output file, .h5>
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

import adc64format

input_file = sys.argv[-2]
output_file = sys.argv[-1]

WRITE_BUFFER = 1024-1

print('''
*************
ADC64 -> HDF5
*************
''', end='')

print(f'opening {input_file}')
with adc64format.ADC64Reader(input_file) as reader:
    print('success!')

    print('reading first block')
    _, nbytes, test_event = adc64format.parse_chunk(reader.streams[0])
    nchannels = len(test_event['data'])
    nsamples = test_event['data'].dtype['voltage'].shape[-1]
    print(f'block size is {nbytes}B, with {nchannels} channels and {nsamples} samples')

    print('checking file length')
    size = reader.streams[0].seek(0, 2)
    reader.streams[0].seek(0, 0)
    chunk_size = adc64format.chunk_size(reader.streams[0])
    print(f'file contains {size // chunk_size} events')
    
    print(f'creating {output_file}')
    with h5py.File(output_file, 'w') as fo:
        print('success!')

        print('generating datasets')
        dtypes = adc64format.dtypes.copy()
        dtypes['data'] = adc64format.dtypes['data'](nsamples)
        dtypes['ref'] = np.dtype([('start', 'i4'), ('stop', 'i4')])
        for key in dtypes:
            initial_shape = (size // chunk_size,) if key != 'data' else (nchannels * size // chunk_size,)
            fo.create_dataset(key, initial_shape, maxshape=(None,), dtype=dtypes[key], compression='gzip', shuffle=True)
        dset_keys = list(fo.keys())

        ptr = defaultdict(int)
        data = defaultdict(list)
        print('initial dataset sizes:\n' + '\n'.join([f'  {k}: {fo[k].shape}' for k in dset_keys]))

        print('\n*************\n   CONVERT\n*************')

        with tqdm.tqdm(total=size//chunk_size, unit=' events', smoothing=0) as pbar:
            while True:
                events = reader.next()
                event = events[0] if events is not None else None

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

                            # print(key, 'write', write_data.shape, 'to', write_ptr)
                            fo[key].write_direct(write_data, dest_sel=np.s_[write_ptr-len(write_data):write_ptr])

                            data[key] = list()

                if event is None:
                    break

                pbar.update()

        print('success!')
        for key in dset_keys:
            fo[key].resize((ptr[key],))
        print('final dataset sizes:\n' + '\n'.join([f'  {k}: {fo[k].shape}' for k in dset_keys]))

print('\n*************')
