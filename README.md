# DST80 Cracker
``
pip3 install numpy opencl  
``

time python3 dst80_reverse_aaaa_cte.py 0xC212345678 0x64cfd0 281474976710656  
time python3 dst80_reverse_purebrute.py 0xC212345678 0x64cfd0 281474976710656  


```
(myenv) nirvana@legion:~/dst80-cracker$ time python3 dst80_reverse_purebrute.py 0xC212345678 0x64cfd0 281474976710656  
PID 0 first sig=0x023710, keyl=0x0000000000, keyr=0xffffffffff   
Found verified match @pid=1: keyl=0xccb6190000, keyr=0xffffe64933, sig=0x64cfd0, target=0x64cfd0  
Found verified match @pid=1: keyl=0xa9ab930000, keyr=0xffff6c5456, sig=0x64cfd0, target=0x64cfd0
``  
