import pytest
import numpy as np

from adc64format import ADC64Reader, parse_chunk, dtypes


def test_fail_on_no_args():
    with pytest.raises(AssertionError):
        with ADC64Reader():
            pass


def test_open_file(example_file):
    with ADC64Reader(example_file):
        pass


def test_open_multi_file(example_files):
    with ADC64Reader(*example_files):
        pass


def test_skip_file(example_file):
    with ADC64Reader(example_file) as reader:
        # get first event
        first_event = reader.next()

        # skip some events
        nskip = 300
        reader.skip(nskip)

        # check new event against old
        event = reader.next()

        assert first_event[0]['event'][0]['serial']+nskip+1 == event[0]['event'][0]['serial']

        # skip back to original position
        nskip = -302
        reader.skip(nskip)

        # check new event against old
        event = reader.next()

        assert first_event[0]['event'][0]['serial'] == event[0]['event'][0]['serial']


def test_skip_multi_file(example_files):
    with ADC64Reader(*example_files) as reader:
        # get first event
        first_event = []
        for f in reader.streams:
            first_event.append(parse_chunk(f)[2])

        # skip some events
        nskip = 300
        reader.skip(nskip)

        # check new event against old
        event = []
        for f in reader.streams:
            event.append(parse_chunk(f)[2])

        # check that each file has been advanced
        for ev, first_ev in zip(event, first_event):
            assert first_ev['event']['serial']+nskip+1 == ev['event']['serial']


def test_read_file(example_file):
    i = 0
    batch_size = 7
    max_iterations = 10000
    with ADC64Reader(example_file) as reader:
        while events := reader.next(batch_size):
            # check that we return correct type
            assert isinstance(events, list), 'returned type is incorrect'
            assert len(events) == 1, 'length of returned object is incorrect'

            # check that we get roughly the number of events that we expect
            for key in dtypes:
                assert len(events[0][key]) <= batch_size, 'number of returned events is too large'
                assert len(events[0][key]) > 0, 'number of returned events is 0'

            # check for infinite loop
            i += 1
            if i > max_iterations:
                raise RuntimeError(f'event loop failed to terminate after {max_iterations} iterations')


def test_read_multi(example_files):
    i = 0
    batch_size = 7
    max_iterations = 10000
    with ADC64Reader(*example_files) as reader:
        while events := reader.next(batch_size):
            # check return type
            assert isinstance(events, list), 'returned type is incorrect'

            # check that we get roughly the number of events that we expect
            for event in events:
                for key in dtypes:
                    assert len(event[key]) <= batch_size, 'number of returned events is too large'
                    assert len(event[key]) > 0, 'number of returned events is 0'

            # check that no events are totally empty
            assert not all([all([entry is None for entry in event['header']]) for event in events]), 'all entries in all events are null'

            # check for infinite loop
            i += 1
            if i > max_iterations:
                raise RuntimeError(f'event loop failed to terminate after {max_iterations} iterations')


def test_sync_multi(example_files):
    max_events = 1500
    sync_fraction_tol = 0.99
    with ADC64Reader(*example_files) as reader:
        events = reader.next(max_events)

        # calculate timestamps
        timestamp = np.c_[[[int(entry['tai_s']*1e9 + entry['tai_ns'])
                            if entry is not None else -1
                            for entry in event['time']]
                           for event in events]]

        # skip data up to first sync event
        sync_start = np.argmax(timestamp[0] == 0)
        syncd_timestamp = timestamp[:, sync_start:]
        # calculate the number of well-matched events
        syncd_mask = np.abs(np.diff(syncd_timestamp, axis=0)) < reader.TAI_NS_WINDOW
        syncd_fraction = syncd_mask.sum() / syncd_timestamp.shape[1]

        # check that most events are well matched
        assert syncd_fraction > sync_fraction_tol, f'poor sync fraction: {syncd_fraction*100:0.02f}% ({syncd_mask.sum()}/{syncd_timestamp.shape[1]}), skipped {sync_start}'
