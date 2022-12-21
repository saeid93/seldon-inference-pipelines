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
    if "-" in days and ":" not in days:
        first, end = map(int, days.split("-"))
        first = (first - 1) * seconds_per_day
        end = end * seconds_per_day
        workload_all = get_workload()
        return workload_all[first:end]
    
    elif ":" in days:
        first, second = days.split("-")
        first_day_detail = list(map(int, first.split(":")))
        second_day_detail = list(map(int, second.split(":")))
        first_day = first_day_detail[0]
        second_day = second_day_detail[0]
        first = (first_day - 1) * seconds_per_day
        second = second_day * seconds_per_day
        workload_all = get_workload()
        workload_temp = workload_all[first:second]

        if first_day_detail[1] > 0:
            first_total_seconds = (first_day_detail[1] - 1) * 3600 + 30 * 60
        else:
            first_total_seconds = 0
        
        first_total_seconds += first_day_detail[2] * 60
        if first_day_detail[1] > 0:
            second_total_seconds = (second_day_detail[1] - 1) * 3600 + 30 * 60
        else:
            second_total_seconds = 30 * 60

        second_total_seconds += second_day_detail[2] * 60  + (second_day - first_day) * seconds_per_day
        return workload_temp[first_total_seconds:second_total_seconds]
    
    else:
        workload_all = get_workload()
        first = int(days)
        end = first
        first = first - 1
        first = first *seconds_per_day
        end = end * seconds_per_day
        return workload_all[first:end]

# print(len(twitter_workload_generator("3:2:2-4:5:3")))
# print(len(twitter_workload_generator("3-5")))
# print(len(twitter_workload_generator("3")))

# Use - to show split between days => 3-5
# Use : to indicate you want give an specific time of day => 3:5:3-5:6:7. 3
# without using any of  them, you actually say that you want detail of that specific day =