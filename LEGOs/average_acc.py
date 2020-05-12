import sys
acclst = []
for line in sys.stdin:

    if "attentive" in line:
        print(line)
    
    if "Test" in line:
        acclst.append(float(line.split(' ')[3]))

acclst = acclst[60:73]

print(len(acclst))
print(sum(acclst[-10:])/len(acclst[-10:]))

