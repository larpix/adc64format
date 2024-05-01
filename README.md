adc64format
-----------

This is a script and library to help interfact with the ADC64 format.

To install::

    pip install adc64format

## mpd format (new)
To use interactively or within another python codebase::

    from adc64format import mpd_parse_chunk, mpd_parse_run_start, MPDReader

    input_file = adc64format.MPDReader(input_filename,n_adcs)
    input_file.open()

    # Read run info
    _, nbytes_runinfo, runinfo = adc64format.mpd_parse_run_start(input_file.stream)
    # use the first file for the event reference
    _, chunk_size, test_event = adc64format.mpd_parse_chunk(input_file.stream)
    
    events = input_file.next(batch_size)
    print(events.keys())
    

## adc64 format (old)

To dump the contents of a single ADC64 file to an HDF5 format::

    adc64_to_hdf5.py <input ADC64 file>.data <output HDF5 file>.h5

To use interactively or within another python codebase::

    from adc64format import dtypes, skip_chunks, parse_chunk, ADC64Reader

    # Option 1: Parse a single event from an ordinary file object
    with open('<ADC64 file>.data', 'rb') as f:

        # Load the first event
        chunk = parse_chunk(f)

        # Look at chunk data (as numpy arrays)
        for key in dtypes:
            print(chunk[key])

        # Skip N events
        n = 100
        skip_chunks(f, n)
        chunk = parse_chunk(f)


    # Option 2: Parse events from multiple files and align by timestamp
    with ADC64Reader('<ADC 64 file from ADC 0>.data', '<ADC 64 file from ADC 1>.data', ...) as reader:
        batch_size = 10 # how many events to load on each iteration
        events = reader.next(batch_size)
       
        # get matched events between multiple files
        events_file0, events_file1, ... = events

        # Look at data
        for key in dtypes:
            print(events_file0[key])