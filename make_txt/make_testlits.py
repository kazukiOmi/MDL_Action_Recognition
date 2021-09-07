import csv
import os


def get_id():
    filename = '/mnt/NAS-TVS872XT/dataset/Kinetics/kinetics400/test.csv'
    with open(filename, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        x = [row for row in csvreader]
        print(len(x[0]))
        for num in range(len(x)):
            if num == 1:
                print(x[num])
                print("label:{}, id:{}".format(x[num][0], x[num][1]))


def get_filename():
    file_path = "/mnt/NAS-TVS872XT/dataset/Kinetics400/test"
    files = os.listdir(file_path)
    print(type(files))
    print(len(files))
    print(files[0])
    result = False
    for item in files:
        if "-NKgPEJ_Gsk" in item:
            result = True
            return result
    return False


def main():
    get_filename()


if __name__ == '__main__':
    main()
