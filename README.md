# xsensTools
Tools for working with data from mvn xsens accelerometer suit


#### mvnxParser
Tool to read datafiles with .mvnx extension in python. Can also be used for exportation of mvnx files to different file format (currently .mat and .json is supported). 

* usage: python3 mvnxParser [-h] [-t] [-v] [-l] filenames
    * filenames: filename(s) or directory containing .mvnx files for conversion and export
    * -h, --help:     show this help message and exit
    * -v, --verbose:   increase output verbosity
    * -t, --savetsv:  save data as .tsv table instead of .mat (not implemented yet)
    * -a, --anonimize: use this flag to remove potentially identifying metadata, such as recording date or original filename
    * -m, --merge_metadata: use this flag to merge metadata from recordings in the same directory into a single JSON file
    * -l, --filelist: use this flag when passing multiple filenames (please use comma-separated filenames e.g. 'S01-001.mvnx,S01-002.mvnx,S01-003.mvnx')

