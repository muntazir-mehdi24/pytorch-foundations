import datetime
import csv

def log_message(model, loss, epoch):
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('training_log.csv', mode = 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([time, epoch, loss.item(), model.state_dict()])