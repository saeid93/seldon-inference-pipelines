import os
TESTS_PATH = os.path.dirname(__file__)
seconds_per_day = 30 * 60 + 23 * 3600
# twitter dataset extracted from 
path = os.path.join(TESTS_PATH, "workload.txt")
def get_workload():
    f = open(path, "r")
    data = f.read()
    workload =  data.split(" ")
    return_workload = []
    for i, w in enumerate(workload):
        try:
            return_workload.append(int(w))
        except:
            pass
    f.close()
    return return_workload
def twitter_workload_generator(days):
    if "-" in days:
        first, end = map(int, days.split("-"))
        first = (first - 1) * seconds_per_day
        end = end * seconds_per_day
        workload_all = get_workload()
        return workload_all[first:end]
    
    else:
        workload_all = get_workload()
        first = int(days)
        end = first
        first = first - 1
        first = first *seconds_per_day
        end = end * seconds_per_day
        return workload_all[first:end]
