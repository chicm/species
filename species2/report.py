import os
import pandas as pd
import settings

headers = ['LR', 'Tloss', 'Tacc', 'Tcorrect', 'Vloss', 'Vacc', 'Vcorrect', 'time']
report_data = {}

for header in headers:
    report_data[header] = []

epoch = [0]
file_name = ['']


def report(report_line):
    for index in range(len(headers)):
        report_data[headers[index]].append(report_line[index])

    epoch[0] += 1
    if epoch[0] % 10 == 0:
        df = pd.DataFrame(report_data, columns=headers)
        df.to_csv(settings.RESULT_DIR + os.sep + file_name[0])


'''
for i in range(10):
    report([0.01, 0.2, 0.98, 100, 0.3, 0.99, 10])
report([0.01, 0.2, 0.98, 100, 0.3, 0.99, 10])

print(report_data)

df = pd.DataFrame(report_data, columns=headers)
'''


def report_valid(epoch_loss, epoch_acc, running_corrects):
    report_data['Vloss'].append(epoch_loss)
    report_data['Vacc'].append(epoch_acc)
    report_data['Vcorrect'].append(running_corrects)


def report_train(epoch_loss, epoch_acc, running_corrects):
    report_data['Tloss'].append(epoch_loss)
    report_data['Tacc'].append(epoch_acc)
    report_data['Tcorrect'].append(running_corrects)


def report_time(elapsed):
    print("report_time")
    report_data['time'].append(elapsed)

    # if epoch[0] % 1 == 0:
    #     df = pd.DataFrame(report_data, columns=headers)
    #     df.to_csv(settings.RESULT_DIR + os.sep + "reports.csv")

    df = pd.DataFrame(report_data, columns=headers)
    df.to_csv(settings.RESULT_DIR + os.sep + file_name[0])

    epoch[0] += 1


def report_lr(lr):
    report_data['LR'].append(lr)


def start(since, name):
    file_name[0] = name + '-' + str(since) + '.csv'
