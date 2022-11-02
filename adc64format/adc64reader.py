import numpy as np
from collections import defaultdict

class ADC64Reader(object):
    '''
    Class to help read the raw ADC64 format into numpy arrays, to use::

        from adc64format import ADC64Reader

        batch_size = 64
        with ADC64Reader(<filename>) as reader:
            while (data := reader.next(batch_size)) is not None:
                # do stuff with data
                pass

    Each iteration will return a dict with keys for each of the different
    data blocks in the ADC64 format. The values at each of the keys will be
    a list of the next ``batch_size`` events from the file.

    The class also supports reading and aligning multiple files by their
    timestamp::

        with ADC64Reader(<filename0>, <filename1>, ...) as reader:
            # set the tolerance for event matching
            reader.TIMESTAMP_WINDOW = 5
            while (data := reader.next(batch_size)) is not None:
                # do stuff with data
                pass
    
    In this case, each iteration will return a list of dicts, one per file.
    Missing events will be represented in the dicts with a length 0 array.

    '''

    #: Use this channel and threshold to synchronize multiple files
    SYNC_CHANNEL = 32
    SYNC_THRESHOLD = 1024
    
    #: allow events as long as they are synchronized to within this value
    UNIX_WINDOW = 100 # ms
    TAI_NS_WINDOW = 1000 # ticks

    #: flag to dump *all* data to terminal
    VERBOSE = False

    #: Key and array data types returned in each iteration
    dtypes = dict(
        header=np.dtype([('header','u4'), ('size','u4'), ('unix','u8')]),
        event=np.dtype([('event','u4'), ('size','u4'), ('serial','u4')]),
        device=np.dtype([('serial','u4'), ('id','u1'), ('size','u4')]),
        time=np.dtype([('size','u4'), ('tai_s','u4'), ('tai_ns','u4'), ('flag','u1'), ('bit_mask','u8')]),
        data=lambda nsamples : np.dtype([('channel','u1'), ('size','u4'), ('voltage','i2',(nsamples,))]),
        )
    
    def __init__(self, *filenames):
        assert len(filenames) >= 1, 'At least one filename is needed'
        
        self.filenames = filenames
        self.event = 0
        self._fs = list()
        self._nbytes = list()
        self._next_event = [None] * len(self.filenames)
        self._last_sync = [np.zeros((1,), dtype=self.dtypes['time'][['tai_s','tai_ns']])] * len(self.filenames)

        if self.VERBOSE:
            print('Will load data from:', self.filenames)

    def open(self):
        for file_ in self.filenames:
            self._fs.append(open(file_, 'rb'))
            self._nbytes.append(self._fs[-1].seek(0,2))
            self._fs[-1].seek(0)
            if self.VERBOSE:
                print('File',file_,'opened and contains',self._nbytes[-1],'B')
                
        return self

    def close(self):
        if self._fs:
            for i in range(len(self.filenames)):
                self._fs[i].close()
            self._fs = list()

    def skip(self, nevents, nsamples, nchannels):
        ''' Skips forward in each file by a specified number of events assuming a constant number of samples and readout channels '''
        assert len(self._fs), 'File(s) have not been opened yet!'

        nbytes = (
            16 # header
            + 12 # event payload
            + 8 # device
            + 20 # time
            + (4+8+2*nsamples)*nchannels
            ) * nevents
        
        for f in self._fs:
            f.seek(nbytes,1)

    def next(self, n=1):
        # initialize return value list
        return_value = [defaultdict(list) for _ in self._fs]
        eof = [False] * len(self._fs)

        # mark the stop event
        stop_event = self.event + n

        while True:
            # loop over files
            for i,(f, nbytes) in enumerate(zip(self._fs, self._nbytes)):
                # check for end of file
                if eof[i] or (f.seek(0,1) >= nbytes):
                    if self.VERBOSE:
                        print(f'*** EOF ({i}) ***')
                    eof[i] = True
                    continue
                
                # check if event has already been loaded
                if self._next_event[i] is not None:
                    if self.VERBOSE:
                        print(f'*** ({i}) event already loaded ***')
                    continue
                self._next_event[i] = dict()

                # parse next header
                header = self._parse_header(f)

                # parse next event payload
                event_payload = self._parse_event(f)

                # parse remaining block
                device_payload = self._parse_device(f)

                # parse time block
                time = self._parse_time(f)

                # parse data block
                nblocks = bin(int(time['bit_mask'])).count('1')
                data = self._parse_data(f, n=nblocks)

                # tag sync events and reset timestamps
                is_sync = self._check_sync(data)
                if is_sync:
                    self._last_sync[i] = time[['tai_s','tai_ns']].copy()
                    time['tai_s'] = 0
                    time['tai_ns'] = 0
                else:
                    time['tai_s'] = (
                        time['tai_s'] - self._last_sync[i]['tai_s']
                        if time['tai_ns'] > self._last_sync[i]['tai_ns']
                        else (time['tai_s']-1) - self._last_sync[i]['tai_s']) 
                    time['tai_ns'] = (
                        time['tai_ns'] - self._last_sync[i]['tai_ns']
                        if time['tai_ns'] > self._last_sync[i]['tai_ns']
                        else (1e9+time['tai_ns']) - self._last_sync[i]['tai_ns'])

                # add to next event data
                self._next_event[i]['header'] = header
                self._next_event[i]['event'] = event_payload
                self._next_event[i]['device'] = device_payload
                self._next_event[i]['time'] = time
                self._next_event[i]['data'] = data

            # check if no available events
            if not all([ev is None for ev in self._next_event]):
                tai = [ev['time']['tai_ns'] + 1e9*ev['time']['tai_s'] if ev is not None else np.inf for ev in self._next_event]
                tai = [t if t != 0 else np.inf for t in tai]
                ifirst_file = np.argmin(tai)
                event_unix = self._next_event[ifirst_file]['header']['unix']
                event_tai_ns = self._next_event[ifirst_file]['time']['tai_ns']
            else:
                event_unix = 0
                event_tai_ns = 0
            if self.VERBOSE:
                print(f'Event {self.event} matching on timestamp:', event_unix, event_tai_ns)

            # Move events from next to return data
            for i,ev in enumerate(self._next_event):
                if (
                        ev is not None
                        and (abs(event_unix - ev['header']['unix'].astype(int)) < self.UNIX_WINDOW)
                        and (abs(event_tai_ns - ev['time']['tai_ns'].astype(int)) < self.TAI_NS_WINDOW)):                        
                    for key in self.dtypes:
                        return_value[i][key].append(ev[key])
                    self._next_event[i] = None
                else:
                    for key in self.dtypes:
                        return_value[i][key].append(None)
                            
            if self.VERBOSE:
                print(f'Matched on files:',[i for i in range(len(self._next_event)) if return_value[i]['header'][-1] is not None])

            self.event += 1
            if (self.event == stop_event) or all(eof):
                break
            
        # check for EOF across all files
        if all(eof) and all([header is None for rv in return_value for header in rv['header']]):
            return None

        # return arrays
        if len(self.filenames) == 1:
            return return_value[0]
        return return_value

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()        

    @classmethod
    def _parse_header(cls, f):
        ''' Reads next chunk as a "header" block '''
        time_header = np.frombuffer(f.read(8), dtype='u4')
        unix_timestamp = np.frombuffer(f.read(8), dtype='u8')
        
        arr = np.zeros((1,), dtype=cls.dtypes['header'])
        arr['header'] = time_header[0]
        arr['size'] = time_header[1]
        arr['unix'] = unix_timestamp

        if cls.VERBOSE:
            print('\t', arr, arr.dtype)

        return arr

    @classmethod
    def _parse_event(cls, f):
        ''' Reads next chunk as an "event" block '''
        event_payload = np.frombuffer(f.read(12), dtype='u4')
        
        arr = np.zeros((1,), dtype=cls.dtypes['event'])
        arr['event'] = event_payload[0]
        arr['size'] = event_payload[1]
        arr['serial'] = event_payload[2]

        if cls.VERBOSE:
            print('\t', arr, arr.dtype)
        
        # check magic word
        assert hex(int(arr['event'])) == '0x2a502a50', f'Bad event word ({hex(int(arr["event"]))}), file corrupted?'

        return arr

    @classmethod
    def _parse_device(cls, f):
        ''' Reads next chunk as a "device" block '''
        device_serial_number = np.frombuffer(f.read(4), dtype='u4')
        device_payload_size = np.frombuffer(f.read(3)+b'\x00', dtype='u4')
        device_id = np.frombuffer(f.read(1), dtype='u1')

        arr = np.zeros((1,), dtype=cls.dtypes['device'])
        arr['serial'] = device_serial_number
        arr['id'] = device_id
        arr['size'] = device_payload_size

        if cls.VERBOSE:
            print('\t', arr, arr.dtype)

        return arr

    @classmethod
    def _parse_time(cls, f):
        ''' Reads next chunk as a "time" block '''
        time_payload_size = np.frombuffer(f.read(4), dtype='u4') >> 2
        event_tai_s = np.frombuffer(f.read(4), dtype='u4')
        event_tai_ns = np.frombuffer(f.read(4), dtype='u4')
        event_time_flag = event_tai_ns % 4
        event_tai_ns = event_tai_ns >> 2
        bit_mask = np.frombuffer(f.read(8), dtype='u8')

        arr = np.zeros((1,), dtype=cls.dtypes['time'])
        arr['size'] = time_payload_size
        arr['tai_s'] = event_tai_s
        arr['tai_ns'] = event_tai_ns
        arr['flag'] = event_time_flag
        arr['bit_mask'] = bit_mask

        if cls.VERBOSE:
            print('\t', arr, arr.dtype)

        return arr

    @classmethod
    def _parse_data(cls, f, n=1):
        ''' Reads next chunk(s) as a "data" blocks '''
        arr_list = list()
        i = 0
        while i < n:
            size = np.frombuffer(f.read(3)+b'\x00', dtype='u4') >> 2
            channel = np.frombuffer(f.read(1), dtype='u1')
            f.seek(8,1)
            nsamples = int(2*(size-2))
            wvfm = np.frombuffer(f.read(2*nsamples), dtype='i2')

            arr = np.zeros((1,), dtype=cls.dtypes['data'](nsamples))
            arr['size'] = size
            arr['channel'] = channel
            arr['voltage'][0,::2] = wvfm[1::2]
            arr['voltage'][0,1::2] = wvfm[::2]

            arr_list.append(arr)
            i += 1

        arr = np.concatenate(arr_list)
        if cls.VERBOSE:
            print('\t', arr[['channel','size']], arr.dtype[['channel','size']])

        return arr

    def _check_sync(self, data):
        mask = data['channel'] == self.SYNC_CHANNEL
        mask = data[mask]['voltage'] >= self.SYNC_THRESHOLD
        return bool(np.any(mask))
