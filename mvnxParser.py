import xmltodict
import numpy as np
import pandas as pd
import json
import os
from glob import glob
import scipy.io

class MVNX_File(object):
    """Class for parsing MVNX recording files from the XSens MVN accelerometer system
    usage: myfile = MVNX_File('path/to/file.mvnx')

    Attributes:
    Dictionaries containing relevant data and metadata
    - myfile.data
    - myfile.metadata

    For convenience, all keys in above dictionaries are also made directly accesible. e.g.:
    - myfile.acceleration
    - myfile.framerate
    - myfile.data_keys
    - ...

    """

    def __init__(self, filename, verbose=True):
        if verbose:
            print('reading file: ' + filename + ' ..')

        # Read file
        self.xml_dict = self.read_mvnx(filename)

        # Load metadata
        self.metadata = self.load_metadata()
        for item in self.metadata.items():
            setattr(self, *item)

        # Convert xml ordered dict to useful data dictionary
        self.data = self.frames_to_dict()
        for item in self.data.items():
            setattr(self, *item)

    #         if verbose:
    #             print('succesfully loaded')

    def read_mvnx(self, filename):
        # Open file and load as ordered dict
        with open(filename) as f:
            xml = f.read()
            xml_dict = xmltodict.parse(xml)
        return xml_dict

    def load_metadata(self, root=None):
        if root is None:
            root = self.xml_dict

        metadata = {}
        # Version and comments
        metadata['mvn'] = {key[1:]: root['mvnx']['mvn'][key] for key in root['mvnx']['mvn'].keys()}
        metadata['comments'] = root['mvnx']['comment']

        # Recording details
        child = root['mvnx']['subject']
        metadata['framerate'] = float(child['@frameRate'])
        metadata['subject'] = {key[1:]: child[key] for key in child.keys() if key.startswith('@')}
        metadata['segments'] = [segment['@label'] for segment in child['segments']['segment']]
        metadata['sensors'] = [sensor['@label'] for sensor in child['sensors']['sensor']]

        if child['footContactDefinition']:
            metadata['footContactsDefinition'] = [item['@index'] + ': ' + item['@label'] for item in
                                                  child['footContactDefinition']['contactDefinition']]

        child = child['frames']['frame']
        metadata['n_frames'] = len(child)
        metadata['data_keys'] = [key for key in child[-1].keys() if not key.startswith('@')]
        return metadata

    def parse_frame(self, frame, key, root=None):
        """--To parse XML text in numpy array of correct shape--"""
        if root is None:
            root = self.xml_dict

        child = root['mvnx']['subject']['frames']['frame']
        exceptions_4D = {'orientation', 'footContacts', 'sensorOrientation'}
        out = np.array(child[frame][key].split(), dtype=float)
        return out.reshape(-1, 3) if not key in exceptions_4D else out.reshape(-1, 4)

    def frames_to_dict(self, data_keys=None, n_frames=None):
        if data_keys is None:
            data_keys = self.metadata['data_keys']

        if n_frames is None:
            n_frames = self.metadata['n_frames']

        # Parse last frame and evaluate what data is stored in MVNX-file
        data = {}
        for key in data_keys:
            d_shape = self.parse_frame(frame=-1, key=key).shape
            data[key] = np.zeros((n_frames,) + d_shape)

        # Parse all frames, and store converted dict in self.data
        for frame in range(n_frames):
            for key in data_keys:
                try:
                    data[key][frame] = self.parse_frame(frame, key)
                except KeyError:
                    continue

        return data


def export_mvnx(filenames=None, save_type='mat', verbose=True):
    """ Function for exporting mnvx data to different format.
    - Input: directory, filename or comma-separated list of filenames (type=str)
    - Default behavior: data is saved as .mat format and metadata as .json file.
      (set save_type='tsv' to save data as table in .tsv file.)
    """
    # Make sure that input type is correct (list of .mvnx filenames, or directory)
    filenames = filenames if isinstance(filenames,list) else [filenames]
    remove = []
    for fn in filenames:
        if not fn:
            print('Error: no filenames or directories were specified')
            return
        elif not (os.path.exists(fn) or os.path.exists(fn+'.mvnx')):
            print('Error: no such file or directory ', fn)
            remove.append(fn)
            if ',' in fn:
                print("please use flag -l if providing multiple filenames as list. (See mvnxParser -h for all options)")
        elif fn.endswith('.mvn'):
            print('Error: could not convert: ', fn)
            print('Cannot convert type .mvn, please convert to .mvnx in MVN-studio first!')
            remove.append(fn)
        elif os.path.isdir(fn):
            filenames = glob(fn+'/*.mvnx')
            print('converting all files in directory: ', fn)
            return export_mvnx(filenames)
        elif fn.endswith('.mvnx'):
            fn = fn[:-5]

    # List valid filnames (and drop the file-extensions if present)
    valid = [fn for fn in filenames if (not fn.endswith('.mvnx') and not fn in remove)]
    valid += [fn[:-5] for fn in filenames if (fn.endswith('.mvnx') and not fn in remove)]
    filenames = valid
    print('Found %d valid filenames for conversion' %len(filenames))
    if not filenames:
        return


    # Do the actual conversion
    for fn in filenames:
        infile = MVNX_File(fn+'.mvnx', verbose=verbose)
        with open(fn+'.json', 'w') as outfile:
            json.dump({'metadata':infile.metadata},outfile)
        if save_type == 'mat':
            scipy.io.savemat(fn+'.mat', infile.data)
        elif save_type == 'tsv':
            raise NotImplementedError('conversion to tsv is not supported yet')
        if verbose:
            print('successfully converted '+ fn +'.mvnx')

    if remove and verbose:
        print('Did not convert the following items:', remove)
    return

if __name__ == '__main__':
    """ When run independently (i.e. when not called from different code), this script is used for
    exporting .mvnx files to different file format (.mat or .tsv). See export_mvnx() function for details. """


    import argparse
    import time

    # To calculate the time of execution
    start_time = time.time()

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=str,
                        help="filename or directory containing .mvnx files for export/conversion")
    parser.add_argument("-t", "--savetsv", action="store_true",
                        help="save data as .tsv instead of .mat")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-l", "--filelist", action="store_true",
                        help="use this flag when passing multiple filenames (please use comma-separated filenames e.g. 'S01-001.mvnx,S01-002.mvnx,S01-003.mvnx')  ")
    args = parser.parse_args()

    # Unparse arguments
    filenames = args.filenames if not args.filelist else args.filenames.split(sep=',')
    save_type = 'mat' if not args.savetsv else 'tsv'
    verbose = args.verbose

    # Export mvnx filenames
    export_mvnx(filenames=filenames, save_type=save_type, verbose=verbose)
    print('Finished in %3.2f minutes' % ((time.time() - start_time) / 60))
