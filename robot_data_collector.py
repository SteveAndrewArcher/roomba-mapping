
import serial
import time
import struct

def sendCommand(command):
    global s
    cmd = ""
    for v in command.split():
        cmd += chr(int(v))
    try:
        if s is not None:
            s.write(cmd)
        else:
            print "Not connected."
    except serial.SerialException:
        print "Lost connection"
        print('Uh-oh', "Lost connection to the robot!")
   

START = '128'
BAUD = '129'
CONTROL = '130'
FULL_MODE_COMMAND = '132'
STREAM = '148'
DEMO = '136'

data = []
     
s = serial.Serial('COM6', baudrate=115200, timeout=10)
print(s.name)

print(s.baudrate)
print(s.bytesize)

sendCommand(START)
sendCommand(CONTROL)
sendCommand(FULL_MODE_COMMAND)


#start stream of sensor data
sendCommand(STREAM)
sendCommand('5') #number of packets to send
sendCommand('7') #bumps and wheeldrops packet
sendCommand('8') #wall sensor
sendCommand('19') #distance
sendCommand('20') #angle in degrees
sendCommand('45')



#start start normal cover
sendCommand(DEMO)
sendCommand('0')


##  expected response each 15ms
##  0   |   1   |    2    |   3   |    4    |   5   |    6     |     7      |     8     |     9    |    10      |      11   |   12     |  13    |   14
##header|n-bytes|packetID7|packet7|packetID8|packet8|packetID19|packet19high|packet19low|packetID20|packet20high|packet20low|packetID45|packet45|checksum           
           
begin = time.time()
while time.time()-begin < 45: #time to run and gather data
    data.extend(s.read(256))


#stop robot
sendCommand(DEMO)
sendCommand('255')


#pause stream of sensor data
sendCommand('150')
sendCommand('0')

s.close()


#line format to write to file:
#bump wall distance angle
f = open("roombadata.txt", "w")
for packet in range(15,len(data)-14): #note: starting at index 13, first packet often has bad values (maybe showing state change from previous execution?)
    if ord(data[packet]) == 19 and ord(data[packet+1]) == 12 and sum(ord(x) for x in data[packet:packet+15])&0xFF==0:
        #packet is the start of a good message
        bump = ord(data[packet+3])
        wall = ord(data[packet+5])
        distance = struct.unpack(">h", data[packet+7]+data[packet+8])[0]
        angle = struct.unpack(">h", data[packet+10]+data[packet+11])[0]
        ir = ord(data[packet+13])
        print >> f, "%f %f %f %f %f" % (bump,wall,distance,angle,ir)
f.close()      
            