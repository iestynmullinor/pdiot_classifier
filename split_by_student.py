
import os
import random

DIRECTORY = "./all_respeck"

def get_prefixes():
    prefixes = set()
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(".csv"):
            prefix = filename.split("_")[0]
            prefixes.add(prefix)
    return prefixes

def split_data(students_in_test_set, students_in_dev_set):
    prefixes = list(get_prefixes())
    random.shuffle(prefixes)
    test_set = prefixes[:students_in_test_set]
    dev_set = prefixes[students_in_test_set:students_in_test_set+students_in_dev_set]
    train_set = prefixes[students_in_test_set+students_in_dev_set:]
    train_files = []
    dev_files = []
    test_files = []
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(".csv"):
            prefix = filename.split("_")[0]
            if prefix in train_set:
                train_files.append(filename)
            elif prefix in dev_set:
                dev_files.append(filename)
            elif prefix in test_set:
                test_files.append(filename)
            
    students_in_train_set = set([filename.split("_")[0] for filename in train_files])
    students_in_dev_set = set([filename.split("_")[0] for filename in dev_files])
    students_in_test_set = set([filename.split("_")[0] for filename in test_files])

    students = {"Train Set": students_in_train_set, "Dev Set": students_in_dev_set, "Test Set": students_in_test_set}

    for set_name, student_list in students.items():
        print(f"{set_name}: {', '.join(student_list)}")

    return train_files, dev_files, test_files

