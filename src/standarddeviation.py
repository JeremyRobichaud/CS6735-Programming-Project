import math


files = ["./logs/MOST.log", "./logs/REST.log"]

for f in files:

    test_name = ""
    database_name = ""
    accuracies = []
    average = 0
    n = 0

    for line in open(f, 'r'):
        if "Testing " in line:
            test_name = line.split("Testing")[1].split("on")[0].strip()
            database_name = line.split("data")[1].replace("\\", "").replace(".", "").strip()

        elif "[INFO]:" in line:

            average = float(line.split(" = ")[1])

            assert test_name and database_name and accuracies and average and n

            stn_dev = [float(a-average) for a in accuracies]
            stn_dev = [pow(a, 2) for a in stn_dev]
            stn_dev = sum(stn_dev)
            stn_dev = stn_dev / n
            stn_dev = math.sqrt(stn_dev)

            print(f"The Average of {test_name} on {database_name} is {average} and with a Standard Deviation of {stn_dev}")

            test_name, database_name, accuracies, average, n = "", "", [], 0, 0

        elif "[Iteration " in line:
            continue

        elif "[Fold " in line:
            n += 1
            cur_acc = float(line.split("found ")[1])
            accuracies.append(cur_acc)
