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

nsamples = int(sys.argv[-3])
input_file = sys.argv[-2]
output_file = sys.argv[-1]

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


print(f'Opening input file {input_file}...', end='')
with open(input_file, 'rb') as fi:

    nbytes = fi.seek(0,2)
    fi.seek(0)
    print(f'{nbytes//1024//1024}MB!')

    print(f'Creating new file at {output_file}...', end='')
    with h5py.File(output_file, 'w') as fo:

        fo.create_dataset('header', (0,), maxshape=(None,), dtype=header_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('event', (0,), maxshape=(None,), dtype=event_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('device', (0,), maxshape=(None,), dtype=device_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('time', (0,), maxshape=(None,), dtype=time_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('data', (0,), maxshape=(None,), dtype=data_dtype, compression='gzip', shuffle=True)
        fo.create_dataset('ref', (0,), maxshape=(None,), dtype=ref_dtype, compression='gzip', shuffle=True)
        dset_keys = list(fo.keys())

        ptr = dict()
        for key in dset_keys:
            ptr[key] = 0

        print('Done!')

        with tqdm.tqdm(total=nbytes, unit='MB', unit_scale=1/1024/1024, smoothing=0) as pbar:

            while True:
                    here = fi.seek(0,1)
                    if here >= nbytes:
                        break
                    
                    # read time header
                    header_data = np.zeros((1,), dtype=header_dtype)
                    time_header = np.frombuffer(fi.read(8), dtype='u4')
                    unix_timestamp = np.frombuffer(fi.read(8), dtype='u8')
                    header_data['header'] = time_header[0]
                    header_data['size'] = time_header[1]
                    header_data['unix'] = unix_timestamp
                    ptr['header'] += 1

                    # read event payload
                    event_data = np.zeros((1,), dtype=event_dtype)
                    event_payload = np.frombuffer(fi.read(12), dtype='u4')
                    event_data['event'] = event_payload[0]
                    event_data['size'] = event_payload[1]
                    event_data['serial'] = event_payload[2]
                    ptr['event'] += 1
                    assert hex(int(event_data['event'])) == '0x2a502a50', f'Bad event word ({hex(int(event_data["event"]))}), file corrupted or invalid sample specification'

                    # read device payload
                    device_data = np.zeros((1,), dtype=device_dtype)
                    device_serial_number = np.frombuffer(fi.read(4), dtype='u4')
                    device_payload_size = np.frombuffer(fi.read(3)+b'\x00', dtype='u4')
                    device_id = np.frombuffer(fi.read(1), dtype='u1')
                    device_data['serial'] = device_serial_number
                    device_data['id'] = device_id
                    device_data['size'] = device_payload_size
                    ptr['device'] += 1

                    # read time block
                    time_data = np.zeros((1,), dtype=time_dtype)
                    time_payload_size = np.frombuffer(fi.read(4), dtype='u4') >> 2
                    event_tai_s = np.frombuffer(fi.read(4), dtype='u4')
                    event_tai_ns = np.frombuffer(fi.read(4), dtype='u4')
                    event_time_flag = event_tai_ns % 4
                    event_tai_ns = event_tai_ns >> 2
                    bit_mask = np.frombuffer(fi.read(8), dtype='u8')
                    time_data['size'] = time_payload_size
                    time_data['tai_s'] = event_tai_s
                    time_data['tai_ns'] = event_tai_ns
                    time_data['flag'] = event_time_flag
                    time_data['bit_mask'] = bit_mask
                    ptr['time'] += 1

                    # read data blocks
                    nblocks = bin(int(bit_mask)).count('1')
                    data = np.zeros(nblocks, dtype=data_dtype)
                    for i in range(nblocks):
                        data[i]['size'] = np.frombuffer(fi.read(3)+b'\x00', dtype='u4') >> 2
                        data[i]['channel'] = np.frombuffer(fi.read(1), dtype='u1')
                        fi.seek(8,1)
                        wvfm = np.frombuffer(fi.read(2*nsamples), dtype='i2')
                        data[i]['voltage'][::2] = wvfm[1::2]
                        data[i]['voltage'][1::2] = wvfm[::2]
                    ptr['data'] += nblocks

                    # create index
                    ref_data = np.zeros((1,), dtype=ref_dtype)
                    ref_data['stop'] = ptr['data']
                    ref_data['start'] = ptr['data'] - len(data)
                    ptr['ref'] += 1

                    # write to hdf5 file
                    for key in dset_keys:
                        if len(fo[key]) <= ptr[key]:
                            fo[key].resize((2*ptr[key],))

                    fo['header'].write_direct(header_data, dest_sel=np.s_[ptr['header']-len(header_data):ptr['header']])
                    fo['event'].write_direct(event_data, dest_sel=np.s_[ptr['event']-len(event_data):ptr['event']])
                    fo['device'].write_direct(device_data, dest_sel=np.s_[ptr['device']-len(device_data):ptr['device']])
                    fo['time'].write_direct(time_data, dest_sel=np.s_[ptr['time']-len(time_data):ptr['time']])
                    fo['data'].write_direct(data, dest_sel=np.s_[ptr['data']-len(data):ptr['data']])
                    fo['ref'].write_direct(ref_data, dest_sel=np.s_[ptr['ref']-len(ref_data):ptr['ref']])

                    pbar.update(fi.seek(0,1)-here)

        for key in dset_keys:
            fo[key].resize((ptr[key],))
            print(f'"{key}" final shape of {ptr[key]}')              

print('Conversion complete!')
