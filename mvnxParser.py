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

    def __init__(self, filename, trim=None, verbose=True):

        self.filename = filename
        self.trim = trim

        if verbose:
            print('reading file: ' + filename + ' ..')

        # Read file
        self.xml_dict = self.read_mvnx(filename)
        assert self.xmldict['mvnx']['subject']['sensors'] is not None, 'Failed to locate raw sensor data in MVNX-file. Check export settings in MVN Studio.'
        self.load_frames(trim=trim,verbose=verbose)



    def __len__(self):
        return len(self.xml_dict['mvnx']['subject']['frames']['frame'])


    def load_frames(self,trim=None,verbose=True):

        self.trim = trim
        if self.trim is not None:
            if self.trim[1] == -1:
                self.trim[1] = len(self)
            if verbose:
                print('Attention, trimming/slicing the recording! only loading frames %d to %d' %(self.trim[0],self.trim[1]))

        # Load metadata
        self.metadata = self.load_metadata()
        for item in self.metadata.items():
            setattr(self, *item)

        # Load data (Convert xml ordered dict to useful data dictionary)
        self.data = self.frames_to_dict()
        for item in self.data.items():
            setattr(self, *item)
        return


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
        metadata['framerate']   = float(child['@frameRate'])
        metadata['subject']     = {key[1:]: child[key] for key in child.keys() if key.startswith('@')}
        metadata['segments']    = [segment['@label'] for segment in child['segments']['segment']]
        metadata['sensors']     = [sensor['@label'] for sensor in child['sensors']['sensor']]

        if child['footContactDefinition']:
            metadata['footContactsDefinition'] = [item['@index'] + ': ' + item['@label'] for item in
                                                  child['footContactDefinition']['contactDefinition']]

        child = child['frames']['frame']
        metadata['n_frames']  = len(child) if self.trim is None else self.trim[1] - self.trim[0]
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

    def frames_to_dict(self, data_keys=None):

        n_frames = self.metadata['n_frames']
        frame_range = range(n_frames) if self.trim is None else range(self.trim[0],self.trim[1])
        if data_keys is None:
            data_keys = self.metadata['data_keys']



        # Parse last frame and evaluate what data is stored in MVNX-file
        data = {}
        for key in data_keys:
            d_shape = self.parse_frame(frame=-1, key=key).shape
            data[key] = np.zeros((n_frames,) + d_shape)

        # Parse all frames, and store converted dict in self.data
        for frame in frame_range:
            store_idx = frame if self.trim is None else frame - self.trim[0]
            for key in data_keys:
                try:
                    data[key][store_idx] = self.parse_frame(frame, key)
                except KeyError:
                    continue

        return data


def mvnx2df(mvnxFile):
    """
    Creates a Pandas Dataframe containing the data and meta-data obtained from mvnxFile.
    ---Input---
    mvnxFile: MVNX_File object with data and metadata from xsens mvnx recording (see https://github.com/deruyter92/xsensTools)

    ---Output---
    dataframe_MVNX: a Pandas.DataFrame object with named columns.

    """

    # Put arrays in a Pandas DataFrame using the provided labels
    dataframe_MVNX = pd.DataFrame()
    dataframe_MVNX['time'] = np.arange(n_frames) / framerate # time in seconds

    for i, segment in enumerate(segments):  # Put segment data in dataframe (estimation of anatomical landmarks)
        for description, data_array in [('_pos', mvnxFile.position),
                                        ('_vel', mvnxFile.velocity),
                                        ('_acc', mvnxFile.acceleration),
                                        ('_angVel', mvnxFile.angularVelocity),
                                        ('_angAcc', angularAcceleration)]:
            dataframe_MVNX[segment + description + '_x'] = data_array[:, i, 0]
            dataframe_MVNX[segment + description + '_y'] = data_array[:, i, 1]
            dataframe_MVNX[segment + description + '_z'] = data_array[:, i, 2]

    for i, sensor in enumerate(sensors):  # Put sensor data in dataframe (raw data)
        for description, data_array in [('_senFreeAcc', mvnxFile.sensorFreeAcceleration),
                                        ('_senMagFld', mvnxFile.sensorMagneticField)]:
            dataframe_MVNX[sensor + description + '_x'] = data_array[:, i, 0]
            dataframe_MVNX[sensor + description + '_y'] = data_array[:, i, 1]
            dataframe_MVNX[sensor + description + '_z'] = data_array[:, i, 2]

    for description, data_array in [('posCOM', mvnxFile.centerOfMass)]:
        dataframe_MVNX[description + '_x'] = data_array[:, 0, 0]
        dataframe_MVNX[description + '_y'] = data_array[:, 0, 1]
        dataframe_MVNX[description + '_z'] = data_array[:, 0, 2]

    for description, data_array in [('footCon', mvnxFile.footContacts)]:
        dataframe_MVNX[description + '_leftFootHeel']   = data_array[:, 0, 0]
        dataframe_MVNX[description + '_leftFootToe']    = data_array[:, 0, 1]
        dataframe_MVNX[description + '_rightFootHeel']  = data_array[:, 0, 2]
        dataframe_MVNX[description + '_rightFootToe']   = data_array[:, 0, 3]
    return dataframe_MVNX



def export_mvnx(filenames=None, save_type='mat', verbose=False, anonimize=False, merge_metadata=False):
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
            filenames = sorted(glob(fn+'/*.mvnx'))
            print('converting all files in directory: ', fn)
            return export_mvnx(filenames, save_type, verbose, anonimize, merge_metadata)
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

        if anonimize:
            infile.metadata['subject']['recDate']                 = '[REMOVED]'
            infile.metadata['subject']['recDateMSecsSinceEpoch']  = '[REMOVED]'
            infile.metadata['subject']['originalFilename']        = '[REMOVED]'
        with open(fn+'.json', 'w') as outfile:
            json.dump({'metadata':infile.metadata},outfile)
        if save_type == 'mat':
            scipy.io.savemat(fn+'.mat', infile.data)
        elif save_type == 'tsv':
            raise NotImplementedError('conversion to tsv is not supported yet')
        if verbose:
            print('successfully converted '+ fn +'.mvnx')

    if merge_metadata:
        directories = {os.path.split(fn)[0] for fn in filenames}
        for root in directories:
            files = [fn + '.json' for fn in filenames if root == os.path.split(fn)[0]]
            merged = []
            for fn in files:
                with open(fn,'r') as file:
                    merged.append((fn,json.load(file)))
                os.remove(fn)

            with open(os.path.join(root,'Metadata.json'),'w') as file:
                json.dump({os.path.split(fn)[1][:-5]:file['metadata'] for fn,file in merged},file)

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
    parser.add_argument("-tsv", "--savetsv", action="store_true",
                        help="save data as .tsv instead of .mat")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-l", "--filelist", action="store_true",
                        help="use this flag when passing multiple filenames (please use comma-separated filenames e.g. 'S01-001.mvnx,S01-002.mvnx,S01-003.mvnx')  ")
    parser.add_argument("-a", "--anonimize", action="store_true",
                        help="use this flag to remove potentially identifying metadata, such as recording date or original filename")
    parser.add_argument("-m", "--merge_metadata", action="store_true",
                        help="use this flag to merge metadata from recordings in the same directory into a single JSON file")
    args = parser.parse_args()

    # Unparse arguments
    filenames   = args.filenames if not args.filelist else args.filenames.split(sep=',')
    save_type   = 'mat' if not args.savetsv else 'tsv'
    verbose     = args.verbose
    anonimize   = args.anonimize
    merge       = args.merge_metadata

    # Export mvnx filenames
    export_mvnx(filenames=filenames, save_type=save_type, verbose=verbose, anonimize=anonimize,merge_metadata=merge)
    print('Finished in %3.2f minutes' % ((time.time() - start_time) / 60))
