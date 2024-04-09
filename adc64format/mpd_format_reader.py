import numpy as np
from collections import defaultdict

VERBOSE = False

#: Key and array data types returned in each iteration
dtypes = dict(
    run_start=np.dtype([('run_nr', 'u4'), ('size', 'u4')]),
    header=np.dtype([('sync', 'u4'), ('size', 'u4')]),
    event=np.dtype([('event', 'u4'), ('n_dev', 'u4')]),
    device=np.dtype([('serial', 'u4'), ('id', 'u1'), ('size', 'u4')]),
    time=np.dtype([('size', 'u4'), ('tai_s', 'u4'), ('tai_ns', 'u4'), ('flag', 'u1'), ('bit_mask', 'u8')]),
    data=lambda nsamples: np.dtype([('channel', 'u1'), ('size', 'u4'), ('voltage', 'i2', (nsamples,))]),
)

def mpd_parse_run_start(f):
    if VERBOSE:
        print("Start_run_block")
    
    header = np.frombuffer(f.read(8), dtype='u4') #Read TLV header
    assert hex(int(header[0])) == '0x72617453', 'Run start information not found.'
    arr = np.zeros((1,), dtype=dtypes['run_start'])
    size = header[1]
    if VERBOSE:
        print("   sync0: ",hex(int(header[0])))
        print("   size0: ",int(size))
    payload1 = np.frombuffer(f.read(12), dtype='u4') #Read Run number record
    assert hex(int(payload1[0])) == '0x236e7552', 'Run number record not found'
    arr['run_nr'] = payload1[2]
    if VERBOSE:
        print("   run_nr_size: ", payload1[1])
        print("   Run number: ",payload1[2])

    payload2 = np.frombuffer(f.read(8), dtype='u4') #Read Run index record header (only header!!)
    assert hex(int(payload2[0])) == '0x78646e49', 'Run Index record not found'
    if VERBOSE:
        print("   sync3: ",hex(int(payload2[0])))
        print("   size3: ",payload2[1])
    if not payload2[1]:
        print("   Run Index Record empty")
    else:
        payload3 = np.frombuffer(f.read(int(payload2[1])), dtype='u4') #Read Run index record payload
        #for i in range(int(payload2[1])):
    #    print("i=")
    payload4 = np.frombuffer(f.read(8), dtype='u4')
    if VERBOSE:
        print("   pay4: ",hex(int(payload4[0])))
        print("   size4: ", payload4[1])
        print("   Dump JSON block")
    np.frombuffer(f.read(int(payload4[1])), dtype='u4')

    return f.seek(0,1), 36+int(payload2[1])+int(payload4[1]), arr

def mpd_parse_header(f):
    '''
    Reads next block as a "header" block and advances stream position

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    header = np.frombuffer(f.read(8), dtype='u4')

    if hex(int(header[0])) == '0x4e4f534a':
        np.frombuffer(f.read(int(header[1])), dtype='u4') # Dump JSON
        header = np.frombuffer(f.read(8), dtype='u4')

    assert hex(int(header[0])) == '0x2a50d5af', 'Event header not found'

    arr = np.zeros((1,), dtype=dtypes['header'])
    arr['sync'] = header[0]
    arr['size'] = header[1]

    if VERBOSE:
        print("Event header sync: ", hex(int(arr['sync'])))
        print("Event size: ",arr['size'])

    return f.seek(0, 1), 8, arr


def mpd_parse_event(f,size):
    '''
    Reads next block as an "event" block and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    event_number = np.frombuffer(f.read(4), dtype='u4')
    f.seek(0,1)
    arr = np.zeros((1,), dtype=dtypes['event'])
    arr['event'] = int(event_number)

    if VERBOSE:
        print("Event number: ", arr['event'])

    device_payload = []
    time_payload = []
    data_payload = []
    nb_load = 4

    while nb_load < size:
        if VERBOSE:
            print("start read device #",len(device_payload))
        _,nb ,next_device, next_time, next_data  = mpd_parse_device(f)
        nb_load += nb
        device_payload.append(next_device)
        time_payload.append(next_time)
        data_payload.append(next_data)
        if VERBOSE:
            print("done read device #",len(device_payload)-1)
    
    arr["n_dev"] = len(device_payload)

    return f.seek(0, 1), size, arr, device_payload, time_payload, data_payload

    
def mpd_parse_device(f):
    '''
    Reads next block as a "device" block and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    device_serial_number = np.frombuffer(f.read(4), dtype='u4')
    device_payload_size = np.frombuffer(f.read(3)+b'\x00', dtype='u4')
    device_id = np.frombuffer(f.read(1), dtype='u1')
    if VERBOSE:
        print("   Dev serial: ",device_serial_number)
        print("   Dev ID",hex(int(device_id)))
        print("   Dev payload size", device_payload_size)
    f.seek(0,1)
    if VERBOSE:
        print("   read time block")
    _, time_nb, dev_time_payload = mpd_parse_time(f)
    if VERBOSE:
        print("   read data block")
    _, data_nb, dev_data_payload = mpd_parse_data(f,bin(int(dev_time_payload['bit_mask'])).count('1'))

    arr = np.zeros((1,), dtype=dtypes['device'])
    arr['serial'] = device_serial_number
    arr['id'] = device_id
    arr['size'] = device_payload_size

    return f.seek(0, 1), 8 + int(device_payload_size), arr, dev_time_payload, dev_data_payload


def mpd_parse_data(f, n=1):
    '''
    Reads next block(s) as a "data" blocks and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''

    # peek at first block to determine output array shape
    size = np.frombuffer(f.read(3)+b'\x00', dtype='u4')
    subtype = size % 2
    assert subtype==1, 'Expected data mstream, received different subtype'
    size = size >> 2
    f.seek(-3, 1)
    nsamples = int(2 * (size - 2))
    arr = np.zeros((n,), dtype=dtypes['data'](nsamples))
    
    nbytes = 0
    i = 0
    for i in range(n):
        arr[i]['size'] =  np.frombuffer(f.read(3)+b'\x00', dtype='u4') >> 2
        arr[i]['channel'] = np.frombuffer(f.read(1), dtype='u1')
        f.seek(8, 1)
        # waveform is indexed in a funny way, so we need to swap every other sample
        arr[i]['voltage'] = np.frombuffer(f.read(2 * nsamples), dtype='i2').reshape(-1, 2)[:, ::-1].ravel()
        nbytes += 12 + 2 * nsamples
    return f.seek(0, 1), nbytes, arr


def mpd_parse_time(f):
    '''
    Reads next block as a "time" block and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    time_payload_size = np.frombuffer(f.read(4), dtype='u4')
    subtype = time_payload_size % 2
    assert subtype==0, 'Expected time mstream, received different subtype'
    time_payload_size = time_payload_size >> 2
    event_tai_s = np.frombuffer(f.read(4), dtype='u4')
    event_tai_ns = np.frombuffer(f.read(4), dtype='u4')
    event_time_flag = event_tai_ns % 4
    event_tai_ns = event_tai_ns >> 2
    bit_mask = np.frombuffer(f.read(8), dtype='u8')

    arr = np.zeros((1,), dtype=dtypes['time'])
    arr['size'] = time_payload_size
    arr['tai_s'] = event_tai_s
    arr['tai_ns'] = event_tai_ns
    arr['flag'] = event_time_flag
    arr['bit_mask'] = bit_mask

    return f.seek(0, 1), 20, arr

def mpd_parse_run_stop(f):
    if VERBOSE:
        print("Stop_run_block")
    
    header = np.frombuffer(f.read(8), dtype='u4') #Read TLV header
    assert hex(int(header[0])) == '0x706f7453', 'Run stop information not found.'
    arr = np.zeros((1,), dtype=dtypes['run_start'])
    size = header[1]
    if VERBOSE:
        print("   sync0: ",hex(int(header[0])))
        print("   size0: ",int(size))
    payload1 = np.frombuffer(f.read(12), dtype='u4') #Read Run number record
    assert hex(int(payload1[0])) == '0x236e7552', 'Run number record not found'
    arr['run_nr'] = payload1[2]
    if VERBOSE:
        print("   run_nr_size: ", payload1[1])
        print("   Run number: ",payload1[2])

    payload2 = np.frombuffer(f.read(8), dtype='u4') #Read Run index record header (only header!!)
    assert hex(int(payload2[0])) == '0x78646e49', 'Run Index record not found'
    if VERBOSE:
        print("   sync3: ",hex(int(payload2[0])))
        print("   size3: ",payload2[1])
    if not payload2[1]:
        print("   Run Index Record empty")
    else:
        payload3 = np.frombuffer(f.read(int(payload2[1])), dtype='u4') #Read Run index record payload

    return f.seek(0,1), 36+int(payload2[1]), arr



def mpd_parse_chunk(f):
    '''
    Read next MPD event into numpy arrays and advances stream position

    :returns: tuple of new stream position, number of bytes read, dictionary of numpy arrays

    '''

    nbytes = 0
    # parse next header
    _, nb, header = mpd_parse_header(f)
    nbytes += nb

    # parse next event payload
    _, nb, event_payload, device_payload, time_payload, data_payload = mpd_parse_event(f,header["size"][0])
    nbytes += nb

    # add to next chunk data
    chunk = dict()
    chunk['header'] = header   #array
    chunk['event'] = event_payload  #array
    chunk['device'] = device_payload #list of arrays
    chunk['time'] = time_payload  #list of arrays
    chunk['data'] = data_payload  #list of arrays
    if VERBOSE:
        print('event dataset shapes:\n' + '\n'.join([f'  {k}: {len(chunk[k])} ' for k in chunk.keys()]))

    return f.seek(0, 1), nbytes, chunk



def mpd_chunk_size(f):
    ''' Look ahead and get the next chunk size in bytes, does not advance forward in the stream '''
    _, nbytes, _ = mpd_parse_chunk(f)    
    f.seek(-nbytes, 1)
    return nbytes


def mpd_skip_chunks(f, nchunks):
    '''
    Skip N chunks, assuming all chunks have the same number of channels and number of samples

    :returns: tuple of new stream position and the change in stream position
    '''
    if nchunks == 0:
        return f.seek(0, 1), 0
    nbytes = mpd_chunk_size(f) * nchunks
    return f.seek(nbytes, 1), nbytes

def mpd_check_eof(f):
    ''' Look ahead and get the next chunk size in bytes, does not advance forward in the stream '''
    header = np.frombuffer(f.read(8), dtype='u4')

    if VERBOSE:
        print("Event header sync: ", hex(int(header[0])))

    f.seek(-8, 1)
    return hex(int(header[0])) == '0x706f7453'


class MPDReader(object):
    '''
    Class to help read the raw MPD format into numpy arrays, to use::

        from adc64format import MPDReader

        batch_size = 64
        with MPDReader(<filename>) as reader:
            while data := reader.next(batch_size):
                # do stuff with data
                pass

    Each iteration will return a list of length one, containing a dict with
    keys for each of the different data blocks in the MPD format. The
    values at each of the keys will be a list of the next ``batch_size``
    chunks from the file.

    When the end of the file is reached, ``next()`` will return ``None``.

    The class also supports reading and aligning multiple files by their
    timestamp::

        with ADC64Reader(<filename0>, <filename1>, ...) as reader:
            # set the tolerance for event matching
            reader.TIMESTAMP_WINDOW = 5
            while data := reader.next(batch_size):
                # do stuff with data
                pass

    In this case, missing/unmatched events will be represented in the dicts with a ``None``.

    '''

    #: allow events as long as they are synchronized to within this value
    UNIX_WINDOW = 100  # ms
    TAI_NS_WINDOW = 1000  # ticks

    #: flag to dump *all* data to terminal
    VERBOSE = False

    def __init__(self, filename, nadc):
        assert len(filename) >= 1, 'At least one filename is needed'

        self.filename = filename
        self.nadc = nadc
        self.chunk = 0
        self.stream = None
        self.runinfo = None
        self._nbyte = -1
        self._next_event = None
        self._last_sync = [np.zeros((1,), dtype=dtypes['time'][['tai_s', 'tai_ns']])] * self.nadc

        if self.VERBOSE:
            print('Will load data from:', self.filename)

    def open(self):
        self.stream = open(self.filename, 'rb')
        self._nbyte = self.stream.seek(0, 2)
        self.stream.seek(0, 0)
        
        if self.VERBOSE:
            print('File', self.filename, 'opened and contains', self._nbyte, 'B')
            print("Run number: ",self.runinfo["run_nr"])
        return self

    def reset(self):
        ''' Return all stream positions to the start of their respective files '''
        if self.stream is None:
            exit()
        self.stream.seek(0, 0)
        self._next_event = None
        self._last_sync = np.zeros_like(self._last_sync)

    def close(self):
        self.stream.close()
        self.stream = None
        for i in range(self.nadc):
            self._last_sync[:] = 0
            self._next_event = None

    def skip(self, nchunks, istream=None):
        ''' Skips around in each file by a specified number of chunks assuming a constant number of samples and readout channels '''
        with self.streams as f:
            assert f is not None, 'File have not been opened yet!'
            mpd_skip_chunks(f, nchunks)

    def next(self, n=1):
        # initialize return value list
        return_value = [defaultdict(list) for _ in range(n)]
        

        # mark the stop chunk
        stop_chunk = self.chunk + n
        
        for i in range(n):
            eof = mpd_check_eof(self.stream)
            # loop over devices
            # check for end of file
            if eof or (self.stream.tell() >= self._nbyte):
                if self.VERBOSE:
                    print(f'*** EOF ***')
                eof = True
                break
            # check if chunk has already been loaded
            if self._next_event is not None:
                if self.VERBOSE:
                    print(f'*** chunk already loaded ***')
                continue
            # parse next file chunk
            _, _, self._next_event = mpd_parse_chunk(self.stream)

            # check if no available events
            if not all([ev is None for ev in self._next_event]):
                # always use valid entries
                tai = [ev['tai_ns'] + int(1e9) * ev['tai_s'] if ev is not None else np.array([np.inf]) for ev in self._next_event["time"]]
                # take sync events after all other entries, but before an invalid entry
                tai = [t if t != 0 else np.array([np.iinfo(t.dtype).max]) for t in tai]
                ifirst_dev = int(np.argmin(tai))
                event_tai_ns = self._next_event['time'][ifirst_dev]['tai_ns']
            else:
                event_tai_ns = 0
            if self.VERBOSE:
                print(f'Chunk {self.chunk} matching on timestamp:', event_tai_ns)

            # Move events from next to return data
            if self._next_event is not None:
                for key in dtypes:
                    if key == 'run_start':
                        continue
                    return_value[i][key] = self._next_event[key]
                    #print("Key: ",key," Len: ",len(return_value[key]))
                self._next_event = None
            else:
                for key in dtypes:
                    if key == 'run_start':
                        continue
                    return_value[i][key] = None

            self.chunk += 1
            if (self.chunk == stop_chunk) or eof:
                #return_value = return_value[:i+1]
                break
        # return arrays
        return return_value

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        print("CLOSED")
        self.close()
