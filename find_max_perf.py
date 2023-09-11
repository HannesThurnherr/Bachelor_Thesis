
import os

def extract_perf_from_txt(filename):
    perf = []
    lines = open(filename).readlines()
    for line in lines:
        if line[0] == "g" and len(line) > 25:
            try:
                perf.append(float(line.split(":")[-1].replace(" ", "").replace("\n", "")))
            except:
                pass

    return perf, len(lines)


txt_perf = []
lines = []
for i in range(100):
    p, l = extract_perf_from_txt("text_logs_4/log_" + str(i) + ".txt")
    txt_perf.append(max(p))
    lines.append(l)

max_idx = 0
for i in range(len(txt_perf)):
    if(txt_perf[i] == max(txt_perf)):
        max_idx = i

for i in range(len(lines)):
    print()
    gen = (lines[i] - 58) / 20
    for j in range(int(gen)):
        print("|", end="")
    print()
    print(i,":", txt_perf[i], "at", int(gen*10), "generations",end="")
    for j in range(69):
        print(" ", end="")
    print("|")
print(max(txt_perf),"accuracy reached in trial",max_idx,"of current run")
run_dirs_count = len([item for item in os.listdir(os.getcwd()) if os.path.isdir(item) and item.startswith('run')])
print(run_dirs_count, "/ 100 runs complete")

