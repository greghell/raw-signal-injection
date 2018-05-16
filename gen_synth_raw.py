# ynth_raw.py
# creates header_template.raw as original header from generator raw file
# inputs -> time in mins (see line 'nTotNumBlocks = int(math.ceil((5.*60.) / (nblocsize/obsnchan/2./2./(abs(obsbw)/obsnchan*1000000.))))')
#                       'logfile.write('chan ' + str(nChan) + ' - Pol#' + str(nPol) + ' - freq0 : ' + str(cenfreq - obsbw'
#                       'MaxDrift = 5*60*30 / (abs(obsbw)*(10**6)/float(obsnchan))      # 5Hz/s * 60s * 30min / BW'
# inputs -> max signal drift in Hz/s
# inputs -> min SNR ('InjSigPar[nPol][1][nChan] = np.random.uniform(-50,20*np.log10(100/ChansGains[nChan]-3),1)')
# maybe assume necessary 3 cycles ABACAD and ask for total duration?

import os
import glob
import sys
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-r', action='store', dest='initial_file', help='initial raw file')
parser.add_argument('-d', action='store', dest='destination', help='destination dataset file names')
parser.add_argument('-t', type=float, action='store', dest='duration', help='total observation duration (ABACAD)')
parser.add_argument('-m', type=float, action='store', dest='max_drift', help='maximum drift rate')

results = parser.parse_args()

# if len(sys.argv) != 5:
        # print 'gen_synth_raw.py raw_file_name dest_name obs_len max_drift_rate\n'
        # print 'obs_len in mins -> total duration of experiment including ABACAD'
        # print 'max_drift_rate in Hz/s'
        # sys.exit()

pathtoraw = os.path.dirname(results.destination)                # makes the directories to create datasets
if pathtoraw and not os.path.isdir(pathtoraw):
        os.makedirs(pathtoraw)

TotObsDuration = float(results.duration)        # total duration of the observation (each data set lasts for TotObsDuration / 6 -> ABACAD)
DurationSet = TotObsDuration / 6.       # duration of a single data set
if DurationSet < 1.:
        print 'single positioning = ' + str(DurationSet) + ' mins'
        print '... a little short for robust analysis -> increase observation duration'
        sys.exit()

fname = results.initial_file
fread = open(fname,'rb')                                # open first file of data set
fheadertemplate = open(pathtoraw+'/header_template.raw','wb')                           # open first file of data set
currline = str(fread.read(80))          # reads first line
fheadertemplate.write(currline)
nHeaderLines = 1

Ra = [0]*2
Dec = [0]*2
Lst = [0]*2
Az = [0]*2
Za = [0]*2
Pktidx = [0]*2
Dropavg = [0]*2
Droptot = [0]*2
Scanrem = [0]*2
Npkt = [0]*2

while currline[0:3] != 'END':           # until reaching end of header
        currline = str(fread.read(80))  # read new header line
        if currline[0:9] == 'OBSFREQ =':        # read cenral frequency
                cenfreq = float(currline[9:])
        if currline[0:9] == 'RA      =':
                Ra[0] = float(currline[9:])
        if currline[0:9] == 'DEC     =':
                Dec[0] = float(currline[9:])
        if currline[0:9] == 'LST     =':
                Lst[0] = float(currline[9:])
        if currline[0:9] == 'AZ      =':
                Az[0] = float(currline[9:])
        if currline[0:9] == 'ZA      =':
                Za[0] = float(currline[9:])
        if currline[0:9] == 'PKTIDX  =':
                Pktidx[0] = float(currline[9:])
        if currline[0:9] == 'DROPAVG =':
                Dropavg[0] = float(currline[9:])
        if currline[0:9] == 'DROPTOT =':
                Droptot[0] = float(currline[9:])
        if currline[0:9] == 'SCANREM =':
                Scanrem[0] = float(currline[9:])
        if currline[0:9] == 'NPKT    =':
                Npkt[0] = float(currline[9:])
        if currline[0:9] == 'DIRECTIO=':        # read directio flag
                ndirectio = float(currline[9:])
        if currline[0:9] == 'BLOCSIZE=':        # read block size
                nblocsize = float(currline[9:])
        if currline[0:9] == 'OBSBW   =':        # read bandwidth
                obsbw = float(currline[9:])
        if currline[0:9] == 'OBSNCHAN=':        # read number of coarse channels
                obsnchan = float(currline[9:])
        nHeaderLines = nHeaderLines + 1
        fheadertemplate.write(currline)
fheadertemplate.close()

nTotNumBlocks = int(math.ceil((TotObsDuration*60.) / (nblocsize/obsnchan/2./2./(abs(obsbw)/obsnchan*1000000.))))        # total number of blocks for complete observation
nTotNumSam = nTotNumBlocks * nblocsize / 4      # total number of samples for complete observation
nBlocksPerFile = [128]*int(math.ceil(nTotNumBlocks/6/128))      # array containing number of blocks per file
nBlocksPerFile[-1] = (nTotNumBlocks/6)%128

nPadd = 0
if ndirectio == 1:
        nPadd = int((math.floor(80.*nHeaderLines/512.)+1)*512 - 80*nHeaderLines)
statinfo = os.stat(fname)
NumBlocs = int(round(statinfo.st_size / (nblocsize + nPadd + 80*nHeaderLines)))

fread.seek(int((NumBlocs-1)*(80*nHeaderLines+nPadd+nblocsize)),0)

currline = str(fread.read(80))          # reads first line
while currline[0:3] != 'END':           # until reaching end of header
        currline = str(fread.read(80))          # read new header line
        if currline[0:9] == 'RA      =':
                Ra[1] = float(currline[9:])
        if currline[0:9] == 'DEC     =':
                Dec[1] = float(currline[9:])
        if currline[0:9] == 'LST     =':
                Lst[1] = float(currline[9:])
        if currline[0:9] == 'AZ      =':
                Az[1] = float(currline[9:])
        if currline[0:9] == 'ZA      =':
                Za[1] = float(currline[9:])
        if currline[0:9] == 'PKTIDX  =':
                Pktidx[1] = float(currline[9:])
        if currline[0:9] == 'DROPAVG =':
                Dropavg[1] = float(currline[9:])
        if currline[0:9] == 'DROPTOT =':
                Droptot[1] = float(currline[9:])
        if currline[0:9] == 'SCANREM =':
                Scanrem[1] = float(currline[9:])
        if currline[0:9] == 'NPKT    =':
                Npkt[1] = float(currline[9:])

fread.close()

RaVal = float((Ra[1]-Ra[0])/NumBlocs)*np.arange(nTotNumBlocks)
DecVal = float((Dec[1]-Dec[0])/NumBlocs)*np.arange(nTotNumBlocks)
LstVal = float((Lst[1]-Lst[0])/NumBlocs)*np.arange(nTotNumBlocks)
AzVal = float((Az[1]-Az[0])/NumBlocs)*np.arange(nTotNumBlocks)
ZaVal = float((Za[1]-Za[0])/NumBlocs)*np.arange(nTotNumBlocks)
PktidxVal = float((Pktidx[1]-Pktidx[0])/NumBlocs)*np.arange(nTotNumBlocks)
DropavgVal = float((Dropavg[1]-Dropavg[0])/NumBlocs)*np.arange(nTotNumBlocks)
DroptotVal = float((Droptot[1]-Droptot[0])/NumBlocs)*np.arange(nTotNumBlocks)
ScanremVal = float((Scanrem[1]-Scanrem[0])/NumBlocs)*np.arange(nTotNumBlocks)
NpktVal = float((Npkt[1]-Npkt[0])/NumBlocs)*np.arange(nTotNumBlocks)

ChansGains =  10*np.exp(-(0.05*pow(np.linspace(-63./2.,63./2.,64),2)/2))+10     # receiver pass band

logfilename = pathtoraw + '/LogInjection.txt'
logfile = open(logfilename,'w')
InjSigPar = np.zeros((2,3,64))

MaxDrift = float(results.max_drift)*60*TotObsDuration / (abs(obsbw)*(10**6)/obsnchan)   # 5Hz/s * 60s * 30min / BW

if MaxDrift >= float(abs(obsbw)*(10**6)/obsnchan):
        print 'Warning : Maximum drift cannot be higher than ' + str(float(abs(obsbw)*(10**6)/obsnchan)/2/TotObsDuration/60) + ' Hz/s'
        sys.exit()

for nChan in range(int(obsnchan)):
        for nPol in range(2):
                InjSigPar[nPol][0][nChan] = np.random.uniform(-0.5+MaxDrift,+0.5-MaxDrift,1)
                InjSigPar[nPol][1][nChan] = np.random.uniform(-50,20*np.log10((3*ChansGains[nChan]+10)/ChansGains[nChan]-3),1)
                InjSigPar[nPol][2][nChan] = np.random.uniform(-MaxDrift,MaxDrift,1)
                logfile.write('chan ' + str(nChan) + ' - Pol#' + str(nPol) + ' - freq0 : ' + str(cenfreq - obsbw/2. + (nChan + (InjSigPar[nPol][0][nChan]+0.5))*obsbw/obsnchan) + ' - drift : ' + str(InjSigPar[nPol][2][nChan] * (abs(obsbw)*(10**6)/float(obsnchan)) / 30. / 60.) + ' - SNR : ' + str(InjSigPar[nPol][1][nChan])+'\n')
logfile.close()



def write_header(output_file,RAval,DECval,LSTval,AZval,ZAval,PKTIDXval,DROPAVGval,DROPTOTval,SCANREMval,NPKTval,SourceName):
        fheadertemplate = open(pathtoraw + '/header_template.raw','rb')
        currline = fheadertemplate.read(80)
        output_file.write(currline)
        nHeaderLines = 1
        while str(currline[0:3]) != 'END':
                currline = fheadertemplate.read(80)
                if str(currline[0:9]) == 'NBITS   =':
                        NewVal = 8
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'DIRECTIO=':
                        NewVal = 1
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'OBSERVER=':
                        NewValStr = 'SETI master'
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'PROJID  =':
                        NewValStr = 'Performance_eval'
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'TELESCOP=':
                        NewValStr = 'Virtual telescope'
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'SRC_NAME=':
                        NewValStr = SourceName  #'Fake source'
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr

                if str(currline[0:9]) == 'RA      =':
                        NewVal = RAval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'DEC     =':
                        NewVal = DECval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'LST     =':
                        NewVal = LSTval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'AZ      =':
                        NewVal = AZval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'ZA      =':
                        NewVal = ZAval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'PKTIDX  =':
                        NewVal = PKTIDXval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'DROPAVG =':
                        NewVal = DROPAVGval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'DROPTOT =':
                        NewVal = DROPTOTval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'SCANREM =':
                        NewVal = SCANREMval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr
                if str(currline[0:9]) == 'NPKT    =':
                        NewVal = NPKTval
                        NewValStr = str(NewVal)
                        if len(NewValStr) > 20:
                                NewValStr = NewValStr[0:20]
                        teststr = currline[0:9] + ' '*(20+1-len(NewValStr)) + NewValStr
                        teststr = teststr + ' '*(80-len(teststr))
                        currline = teststr

                output_file.write(currline)
                nHeaderLines = nHeaderLines + 1
        fheadertemplate.close()
        return nHeaderLines

# build coarse channel filter
# x = np.arange(4096)
# filF = -1.268e-07 * np.power(x,4) + 0.001051*np.power(x,3) - 3.006*np.power(x,2) + 3378*x + 1.109e+06
# filF = filF / np.linalg.norm(filF)
# fil = np.fft.fftshift(np.fft.ifft(filF))


nTotBlockNum = 0

for nOBS in range(6):   # data set number
        for nFile in range(len(nBlocksPerFile)):
                if nFile%2:
                        output_file = open(results.destination+'.OFF.00'+str(int(nOBS/10))+str(nOBS%10)+'.00'+str(int(nFile/10))+str(nFile%10)+'.raw',"wb")
                else:
                        output_file = open(results.destination+'.ON.00'+str(int(nOBS/10))+str(nOBS%10)+'.00'+str(int(nFile/10))+str(nFile%10)+'.raw',"wb")
                for nBlock in range(nBlocksPerFile[nFile]):
                        if nOBS % 2:
                                nHeaderLines = write_header(output_file,RaVal[nBlock],DecVal[nBlock],LstVal[nBlock],AzVal[nBlock],ZaVal[nBlock],PktidxVal[nBlock],DropavgVal[nBlock],DroptotVal[nBlock],ScanremVal[nBlock],NpktVal[nBlock],'off target')
                        else:
                                nHeaderLines = write_header(output_file,RaVal[nBlock],DecVal[nBlock],LstVal[nBlock],AzVal[nBlock],ZaVal[nBlock],PktidxVal[nBlock],DropavgVal[nBlock],DroptotVal[nBlock],ScanremVal[nBlock],NpktVal[nBlock],'fake source')
                        nTotBlockNum = nTotBlockNum+1
                        nPadd = int((math.floor(80.*nHeaderLines/512.)+1)*512 - 80*nHeaderLines)
                        output_file.write(' '*nPadd)
                        for nChan in range(int(obsnchan)):
                                noise_real_pol0 =  np.random.normal(0., ChansGains[nChan], int(nblocsize/obsnchan/4))
                                noise_imag_pol0 =  np.random.normal(0., ChansGains[nChan], int(nblocsize/obsnchan/4))
                                noise_real_pol1 =  np.random.normal(0., ChansGains[nChan], int(nblocsize/obsnchan/4))
                                noise_imag_pol1 =  np.random.normal(0., ChansGains[nChan], int(nblocsize/obsnchan/4))

                                pol0 = np.zeros(int(nblocsize/obsnchan/4),dtype=complex)
                                pol1 = np.zeros(int(nblocsize/obsnchan/4),dtype=complex)

                                if nOBS % 2:
                                        pol0.real = noise_real_pol0
                                        pol0.imag = noise_imag_pol0
                                        pol1.real = noise_real_pol1
                                        pol1.imag = noise_imag_pol1
                                else:
                                        time = np.arange(nblocsize/obsnchan/4) + nTotBlockNum*nblocsize/obsnchan/4

                                        pol0.real = noise_real_pol0 + ChansGains[nChan]*pow(10,InjSigPar[0][1][nChan]/20)*np.cos(2*np.pi*np.multiply(InjSigPar[0][0][nChan] + InjSigPar[0][2][nChan]/nTotNumSam/2*time,time))
                                        pol0.imag = noise_imag_pol0 + ChansGains[nChan]*pow(10,InjSigPar[0][1][nChan]/20)*np.sin(2*np.pi*np.multiply(InjSigPar[0][0][nChan] + InjSigPar[0][2][nChan]/nTotNumSam/2*time,time))
                                        pol1.real = noise_real_pol1 + ChansGains[nChan]*pow(10,InjSigPar[1][1][nChan]/20)*np.cos(2*np.pi*np.multiply(InjSigPar[1][0][nChan] + InjSigPar[1][2][nChan]/nTotNumSam/2*time,time))
                                        pol1.imag = noise_imag_pol1 + ChansGains[nChan]*pow(10,InjSigPar[1][1][nChan]/20)*np.sin(2*np.pi*np.multiply(InjSigPar[1][0][nChan] + InjSigPar[1][2][nChan]/nTotNumSam/2*time,time))


                                # pol0 = np.convolve(pol0,fil)
                                # pol0 = pol0[0:nblocsize/obsnchan/4]
                                # pol1 = np.convolve(pol1,fil)
                                # pol1 = pol1[0:nblocsize/obsnchan/4]

                                noise = np.concatenate((np.atleast_2d(pol0.real),np.atleast_2d(pol0.imag),np.atleast_2d(pol1.real),np.atleast_2d(pol1.imag)),axis=0)
                                noise = np.reshape(noise,(1,int(nblocsize)/int(obsnchan)),order='F')

                                noise = noise.astype(np.int8)
                                noise.tofile(output_file)
                        print "file #" + str(nFile) + " - block #" + str(nBlock)
                output_file.close()