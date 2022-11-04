import subprocess
import os
import pytest
import warnings

from adc64format import dtypes

try:
    import h5py
except Exception as e:
    warn_msg = f'could not load h5py ({e}), skipping script testing'
    warnings.warn(warn_msg)
    pytest.skip(warn_msg, allow_module_level=True)


@pytest.fixture
def hdf5_example_file(example_file, tmp_path):
    output_filename = os.path.join(tmp_path, os.path.basename(example_file) + '.h5')
    command = f'python adc64_to_hdf5.py {example_file} {output_filename}'

    subprocess.run(command.split(), check=True)

    yield output_filename

    os.remove(output_filename)


def test_hdf5_script(hdf5_example_file):
    example_events = 500
    example_samples = 256
    example_channels = 58
    with h5py.File(hdf5_example_file, 'r') as f:
        # check that datasets exist and are correct shape
        for key in dtypes:
            # check for basic datatypes
            assert key in f, f'missing dataset: {key}'
            if key != 'data':
                # check for correct number of events
                assert len(f[key]) == example_events
            else:
                # check for correct waveform shape
                assert f['data'].shape == (example_events * example_channels,)
                assert f['data'].dtype['voltage'].shape == (example_samples,)

        # check that we can match waveforms to events
        assert 'ref' in f, 'missing dataset: ref'
        assert len(f['ref']) == example_events
        for i in range(len(f['ref'])):
            ref = f['ref'][i]
            wvfm = f['data'][ref['start']:ref['stop']]
            assert wvfm.shape == (example_channels,), f'row {i} has invalid number of channels'
