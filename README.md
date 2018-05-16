# raw-signal-injection

# copy_header.py
reads the header of a raw file, and copies it to a new file named 'header_tmp.raw'. That new file is in the same directory as the script (NOT the original file)  
usage:  
copy_header.py fName

# gen_synth_raw.py
usage: gen_synth_raw.py [-h] [-r INITIAL_FILE] [-d DESTINATION] [-t DURATION] [-m MAX_DRIFT]  
  
optional arguments:  
  -h, --help       show this help message and exit  
  -r INITIAL_FILE  initial raw file  
  -d DESTINATION   destination dataset file names  
  -t DURATION      total observation duration (ABACAD)  
  -m MAX_DRIFT     maximum drift rate  
creates a RAW dataset made of 3 pairs of ON-OFF observations containing frequency drifting ETI signals and no RFI  

# gen_synth_raw_rfi.py
usage: gen_synth_raw_rfi.py [-h] [-r INITIAL_FILE] [-d DESTINATION] [-t DURATION] [-m MAX_DRIFT]  
  
optional arguments:  
  -h, --help       show this help message and exit  
  -r INITIAL_FILE  initial raw file  
  -d DESTINATION   destination dataset file names  
  -t DURATION      total observation duration (ABACAD)  
  -m MAX_DRIFT     maximum drift rate  
creates a RAW dataset made of 3 pairs of ON-OFF observations containing frequency drifting ETI signals and non-drifting RFI
