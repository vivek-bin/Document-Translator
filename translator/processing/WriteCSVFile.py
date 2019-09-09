import csv


def writeListToCSVFile(newfilePath,rows):

    with open(newfilePath, "a", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row in rows:
            #print(row)
            writer.writerow(row)