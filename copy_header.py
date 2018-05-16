# python ~/artificial_transmissions/copy_header.py /mnt_blc02/datax2/datax2/dibas.20160315/AGBT16A_999_92/GUPPI/C/blc02_2bit_guppi_57463_12521_HIP32984_0003.0003.raw
# copies a header from a RAW file to use as template in gen_synth_raw.py
# the template is in 'header_tmp.raw'

import sys

fname = sys.argv[1]
fread = open(fname,'rb')
fwrite = open('header_tmp.raw','wb')
currline = fread.read(80)

while str(currline[0:3]) != 'END':
        currline = fread.read(80)
        fwrite.write(currline)

fread.close()
fwrite.close()