import numpy as np
from collections import defaultdict

#: Key and array data types returned in each iteration
dtypes = dict(
    header=np.dtype([('header', 'u4'), ('size', 'u4'), ('unix', 'u8')]),
    event=np.dtype([('event', 'u4'), ('size', 'u4'), ('serial', 'u4')]),
    device=np.dtype([('serial', 'u4'), ('id', 'u1'), ('size', 'u4')]),
    time=np.dtype([('size', 'u4'), ('tai_s', 'u4'), ('tai_ns', 'u4'), ('flag', 'u1'), ('bit_mask', 'u8')]),
    data=lambda nsamples: np.dtype([('channel', 'u1'), ('size', 'u4'), ('voltage', 'i2', (nsamples,))]),
)


def parse_header(f):
    '''
    Reads next block as a "header" block and advances stream position

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    time_header = np.frombuffer(f.read(8), dtype='u4')
    unix_timestamp = np.frombuffer(f.read(8), dtype='u8')

    arr = np.zeros((1,), dtype=dtypes['header'])
    arr['header'] = time_header[0]
    arr['size'] = time_header[1]
    arr['unix'] = unix_timestamp

    return f.seek(0, 1), 16, arr


def parse_event(f):
    '''
    Reads next block as an "event" block and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    event_payload = np.frombuffer(f.read(12), dtype='u4')

    arr = np.zeros((1,), dtype=dtypes['event'])
    arr['event'] = event_payload[0]
    arr['size'] = event_payload[1]
    arr['serial'] = event_payload[2]

    # check magic word
    assert hex(int(arr['event'])) == '0x2a502a50', f'Bad event word ({hex(int(arr["event"]))}), file corrupted?'

    return f.seek(0, 1), 12, arr


def parse_device(f):
    '''
    Reads next block as a "device" block and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    device_serial_number = np.frombuffer(f.read(4), dtype='u4')
    device_payload_size = np.frombuffer(f.read(3)+b'\x00', dtype='u4')
    device_id = np.frombuffer(f.read(1), dtype='u1')

    arr = np.zeros((1,), dtype=dtypes['device'])
    arr['serial'] = device_serial_number
    arr['id'] = device_id
    arr['size'] = device_payload_size

    return f.seek(0, 1), 8, arr


def parse_data(f, n=1):
    '''
    Reads next block(s) as a "data" blocks and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''

    # peek at first block to determine output array shape
    size = np.frombuffer(f.read(3)+b'\x00', dtype='u4') >> 2
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


def parse_time(f):
    '''
    Reads next block as a "time" block and advances stream position 

    :returns: tuple of new stream position, number of bytes read, numpy array
    '''
    time_payload_size = np.frombuffer(f.read(4), dtype='u4') >> 2
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


def parse_chunk(f):
    '''
    Read next ADC64 chunk into numpy arrays and advances stream position

    :returns: tuple of new stream position, number of bytes read, dictionary of numpy arrays

    '''
    nbytes = 0
    # parse next header
    _, nb, header = parse_header(f)
    nbytes += nb

    # parse next event payload
    _, nb, event_payload = parse_event(f)
    nbytes += nb

    # parse remaining block
    _, nb, device_payload = parse_device(f)
    nbytes += nb

    # parse time block
    _, nb, time = parse_time(f)
    nbytes += nb

    # parse data block
    nblocks = bin(int(time['bit_mask'])).count('1')
    _, nb, data = parse_data(f, n=nblocks)
    nbytes += nb

    # add to next chunk data
    chunk = dict()
    chunk['header'] = header
    chunk['event'] = event_payload
    chunk['device'] = device_payload
    chunk['time'] = time
    chunk['data'] = data

    return f.seek(0, 1), nbytes, chunk


def chunk_size(f):
    ''' Look ahead and get the next chunk size in bytes, does not advance forward in the stream '''
    _, nbytes, _ = parse_chunk(f)    
    f.seek(-nbytes, 1)
    return nbytes


def skip_chunks(f, nchunks):
    '''
    Skip N chunks, assuming all chunks have the same number of channels and number of samples

    :returns: tuple of new stream position and the change in stream position
    '''
    if nchunks == 0:
        return f.seek(0, 1), 0
    nbytes = chunk_size(f) * nchunks
    return f.seek(nbytes, 1), nbytes


class ADC64Reader(object):
    '''
    Class to help read the raw ADC64 format into numpy arrays, to use::

        from adc64format import ADC64Reader

        batch_size = 64
        with ADC64Reader(<filename>) as reader:
            while data := reader.next(batch_size):
                # do stuff with data
                pass

    Each iteration will return a list of length one, containing a dict with
    keys for each of the different data blocks in the ADC64 format. The
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

    #: Use this channel and threshold to synchronize multiple files
    SYNC_CHANNEL = 32
    SYNC_THRESHOLD = 1024

    #: allow events as long as they are synchronized to within this value
    UNIX_WINDOW = 100  # ms
    TAI_NS_WINDOW = 1000  # ticks

    #: flag to dump *all* data to terminal
    VERBOSE = False

    def __init__(self, *filenames):
        assert len(filenames) >= 1, 'At least one filename is needed'

        self.filenames = filenames
        self.chunk = 0
        self.streams = [None] * len(self.filenames)
        self._nbytes = [-1] * len(self.filenames)
        self._next_event = [None] * len(self.filenames)
        self._last_sync = [np.zeros((1,), dtype=dtypes['time'][['tai_s', 'tai_ns']])] * len(self.filenames)

        if self.VERBOSE:
            print('Will load data from:', self.filenames)

    def open(self):
        for i,file_ in enumerate(self.filenames):
            self.streams[i] = open(file_, 'rb')
            self._nbytes[i] = self.streams[i].seek(0, 2)
            self.streams[i].seek(0, 0)
            if self.VERBOSE:
                print('File', file_, 'opened and contains', self._nbytes[i], 'B')

        return self

    def reset(self):
        ''' Return all stream positions to the start of their respective files '''
        for i, f in enumerate(self.streams):
            if f is None:
                continue
            f.seek(0, 0)
            self._next_event[i] = None
            self._last_sync[i] = np.zeros_like(self._last_sync[i])

    def close(self):
        for i in range(len(self.filenames)):
            self.streams[i].close()
            self.streams[i] = None
            self._last_sync[i][:] = 0
            self._next_event[i] = None

    def skip(self, nchunks, istream=None):
        ''' Skips around in each file by a specified number of chunks assuming a constant number of samples and readout channels '''
        for f in self.streams if istream is None else [self.streams[istream]]:
            assert f is not None, 'File(s) have not been opened yet!'
            skip_chunks(f, nchunks)

    def next(self, n=1):
        # initialize return value list
        return_value = [defaultdict(list) for _ in self.streams]
        eof = [False] * len(self.streams)

        # mark the stop chunk
        stop_chunk = self.chunk + n

        while True:
            # loop over files
            for i, (f, nbytes) in enumerate(zip(self.streams, self._nbytes)):
                # check for end of file
                if eof[i] or (f.seek(0, 1) >= nbytes):
                    if self.VERBOSE:
                        print(f'*** EOF ({i}) ***')
                    eof[i] = True
                    continue

                # check if chunk has already been loaded
                if self._next_event[i] is not None:
                    if self.VERBOSE:
                        print(f'*** ({i}) chunk already loaded ***')
                    continue

                # parse next file chunk
                _, _, self._next_event[i] = parse_chunk(f)

                # tag sync events and reset timestamps
                self._next_event[i] = self._apply_sync(i, self._next_event[i])

            # check if no available events
            if not all([ev is None for ev in self._next_event]):
                # always use valid entries
                tai = [ev['time']['tai_ns'] + int(1e9) * ev['time']['tai_s'] if ev is not None else np.array([np.inf]) for ev in self._next_event]
                # take sync events after all other entries, but before an invalid entry
                tai = [t if t != 0 else np.array([np.iinfo(t.dtype).max]) for t in tai]
                ifirst_file = int(np.argmin(tai))
                event_unix = self._next_event[ifirst_file]['header']['unix']
                event_tai_ns = self._next_event[ifirst_file]['time']['tai_ns']
            else:
                event_unix = 0
                event_tai_ns = 0
            if self.VERBOSE:
                print(f'Chunk {self.chunk} matching on timestamp:', event_unix, event_tai_ns)

            # Move events from next to return data
            for i, ev in enumerate(self._next_event):
                if (
                        ev is not None
                        and (abs(event_unix - ev['header']['unix'].astype(int)) < self.UNIX_WINDOW)
                        and (abs(event_tai_ns - ev['time']['tai_ns'].astype(int)) < self.TAI_NS_WINDOW)):
                    for key in dtypes:
                        return_value[i][key].append(ev[key])
                    self._next_event[i] = None
                else:
                    for key in dtypes:
                        return_value[i][key].append(None)

            if self.VERBOSE:
                print('Matched on files:', [i for i in range(len(self._next_event)) if return_value[i]['header'][-1] is not None])

            self.chunk += 1
            if (self.chunk == stop_chunk) or all(eof):
                break

        # check for EOF across all files
        if all(eof) and all([header is None for rv in return_value for header in rv['header']]):
            return None

        # return arrays
        return return_value

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _apply_sync(self, ifile, event):
        is_sync = self._check_sync(event['data'])

        if is_sync:
            self._last_sync[ifile] = event['time'][['tai_s', 'tai_ns']].copy()
            event['time']['tai_s'] = 0
            event['time']['tai_ns'] = 0
        else:
            event['time']['tai_s'] = (
                event['time']['tai_s'] - self._last_sync[ifile]['tai_s']
                if event['time']['tai_ns'] > self._last_sync[ifile]['tai_ns']
                else (event['time']['tai_s']-1) - self._last_sync[ifile]['tai_s'])
            event['time']['tai_ns'] = (
                event['time']['tai_ns'] - self._last_sync[ifile]['tai_ns']
                if event['time']['tai_ns'] > self._last_sync[ifile]['tai_ns']
                else (1e9+event['time']['tai_ns']) - self._last_sync[ifile]['tai_ns'])

        return event

    def _check_sync(self, data):
        mask = data['channel'] == self.SYNC_CHANNEL
        mask = data[mask]['voltage'] >= self.SYNC_THRESHOLD
        return bool(np.any(mask))
