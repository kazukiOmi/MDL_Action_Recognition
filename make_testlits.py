import csv


def main():
    filename = '/mnt/NAS-TVS872XT/dataset/Kinetics/kinetics400/test.csv'
    with open(filename, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        x = [row for row in csvreader]
        print(len(x[0]))
        for num in range(len(x)):
            if num == 1:
                print(x[num])
                print("label:{}, id:{}".format(x[num][0], x[num][1]))


if __name__ == '__main__':
    main()
