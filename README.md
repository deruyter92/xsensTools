# xsensTools
Tools for working with data from mvn xsens accelerometer suit


#### mvnxParser
Tool to read datafiles with .mvnx extension in python. Can also be used for exportation of mvnx files to different file format (currently .mat and .json is supported). 

* usage: python3 mvnxParser [-h] [-t] [-v] [-l] filenames
    * filenames: filename(s) or directory containing .mvnx files for conversion and export
    * -h, --help:     show this help message and exit
    * -t, --savetsv:  save data as .tsv instead of .mat (not implemented yet)
    * -v, --verbose:   increase output verbosity
    * -l, --filelist: use this flag when passing multiple filenames (please use comma-separated filenames 
    e.g. 'S01-001.mvnx,S01-002.mvnx,S01-003.mvnx')
