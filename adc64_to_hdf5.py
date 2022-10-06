#!/usr/bin/env python
#
# convert adc64 raw data file format to a representative hdf5 format
#
# requires:
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

nsamples = int(sys.argv[-3])
input_file = sys.argv[-2]
output_file = sys.argv[-1]

WRITE_BUFFER = 10240

# datasets:
#  - time header
#  - event payload
#  - device payload
#  - time block
#  - data block(s)
header_dtype = np.dtype([('header','u4'), ('size','u4'), ('unix','u8')])
event_dtype = np.dtype([('event','u4'), ('size','u4'), ('serial','u4')])
device_dtype = np.dtype([('serial','u4'), ('id','u1'), ('size','u4')])
time_dtype = np.dtype([('size','u4'), ('tai_s','u4'), ('tai_ns','u4'), ('flag','u1'), ('bit_mask','u8')])
data_dtype = np.dtype([('channel','u1'), ('size','u4'), ('voltage','i2',(nsamples,))])
ref_dtype = np.dtype([('start','i4'), ('stop','i4')])


print(f'Opening input file {input_file}...', end=' ')
with open(input_file, 'rb') as fi:

    nbytes = fi.seek(0,2)
    fi.seek(0)
    print(f'{nbytes//1024//1024}MB!')

    print(f'Creating new file at {output_file}...', end=' ')
    with h5py.File(output_file, 'w') as fo:

        fo.create_dataset('header', (0,), maxshape=(None,), dtype=header_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('event', (0,), maxshape=(None,), dtype=event_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('device', (0,), maxshape=(None,), dtype=device_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('time', (0,), maxshape=(None,), dtype=time_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('data', (0,), maxshape=(None,), dtype=data_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('ref', (0,), maxshape=(None,), dtype=ref_dtype, compression='gzip', shuffle=True)
        dset_keys = list(fo.keys())

        ptr = defaultdict(int)
        data = defaultdict(list)
        print('datasets: ' + ', '.join(dset_keys) + '...', end=' ')

        print('Done!')

        with tqdm.tqdm(total=nbytes, unit='B', smoothing=0) as pbar:

            while True:
                    here = fi.seek(0,1)
                    if here < nbytes:
                        # read time header
                        data['header'].append(np.zeros((1,), dtype=header_dtype))
                        time_header = np.frombuffer(fi.read(8), dtype='u4')
                        unix_timestamp = np.frombuffer(fi.read(8), dtype='u8')
                        data['header'][-1]['header'] = time_header[0]
                        data['header'][-1]['size'] = time_header[1]
                        data['header'][-1]['unix'] = unix_timestamp
                        ptr['header'] += 1

                        # read event payload
                        data['event'].append(np.zeros((1,), dtype=event_dtype))
                        event_payload = np.frombuffer(fi.read(12), dtype='u4')
                        data['event'][-1]['event'] = event_payload[0]
                        data['event'][-1]['size'] = event_payload[1]
                        data['event'][-1]['serial'] = event_payload[2]
                        ptr['event'] += 1
                        assert hex(int(data['event'][-1]['event'])) == '0x2a502a50', f'Bad event word ({hex(int(data["event"][-1]["event"]))}), file corrupted or invalid sample specification, first data: {data["data"][0]["size"]}'

                        # read device payload
                        data['device'].append(np.zeros((1,), dtype=device_dtype))
                        device_serial_number = np.frombuffer(fi.read(4), dtype='u4')
                        device_payload_size = np.frombuffer(fi.read(3)+b'\x00', dtype='u4')
                        device_id = np.frombuffer(fi.read(1), dtype='u1')
                        data['device'][-1]['serial'] = device_serial_number
                        data['device'][-1]['id'] = device_id
                        data['device'][-1]['size'] = device_payload_size
                        ptr['device'] += 1

                        # read time block
                        data['time'].append(np.zeros((1,), dtype=time_dtype))
                        time_payload_size = np.frombuffer(fi.read(4), dtype='u4') >> 2
                        event_tai_s = np.frombuffer(fi.read(4), dtype='u4')
                        event_tai_ns = np.frombuffer(fi.read(4), dtype='u4')
                        event_time_flag = event_tai_ns % 4
                        event_tai_ns = event_tai_ns >> 2
                        bit_mask = np.frombuffer(fi.read(8), dtype='u8')
                        data['time'][-1]['size'] = time_payload_size
                        data['time'][-1]['tai_s'] = event_tai_s
                        data['time'][-1]['tai_ns'] = event_tai_ns
                        data['time'][-1]['flag'] = event_time_flag
                        data['time'][-1]['bit_mask'] = bit_mask
                        ptr['time'] += 1

                        # read data blocks
                        nblocks = bin(int(bit_mask)).count('1')
                        data['data'].append(np.zeros(nblocks, dtype=data_dtype))
                        for i in range(nblocks):
                            data['data'][-1][i]['size'] = np.frombuffer(fi.read(3)+b'\x00', dtype='u4') >> 2
                            data['data'][-1][i]['channel'] = np.frombuffer(fi.read(1), dtype='u1')
                            fi.seek(8,1)
                            wvfm = np.frombuffer(fi.read(2*nsamples), dtype='i2')
                            data['data'][-1][i]['voltage'][::2] = wvfm[1::2]
                            data['data'][-1][i]['voltage'][1::2] = wvfm[::2]
                            ptr['data'] += nblocks
                            
                        # create index
                        data['ref'].append(np.zeros((1,), dtype=ref_dtype))
                        data['ref'][-1]['stop'] = ptr['data']
                        data['ref'][-1]['start'] = ptr['data'] - len(data['data'])
                        ptr['ref'] += 1

                    # extend hdf5 file as needed
                    for key in dset_keys:
                        if len(fo[key]) <= ptr[key]:
                            fo[key].resize((2*ptr[key],))

                    # write data
                    for key in dset_keys:
                        if (len(data[key]) > WRITE_BUFFER or here >= nbytes) and (len(data[key]) > 0):
                            write_ptr = ptr[key]
                            write_data = np.concatenate(data[key])

                            fo[key].write_direct(write_data, dest_sel=np.s_[write_ptr-len(write_data):write_ptr])

                            data[key] = list()

                    pbar.update(fi.seek(0,1)-here)

                    if here >= nbytes:
                        break

        for key in dset_keys:
            fo[key].resize((ptr[key],))
            print(f'"{key}" final shape of {ptr[key]}')

print('Conversion complete!')

