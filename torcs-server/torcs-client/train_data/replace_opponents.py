import glob
import re

csv_list = glob.glob('dont_use_these/*.csv')
ignore = 'dont_use_these/'

for csv in csv_list:
    log_file = open(csv, 'r')
    log = log_file.readlines()
    new_filename = 'corrected_' + csv[len(ignore):]
    new_log = open(new_filename, 'w')
    new_header = log[0].replace('OPPONENTS', 'OPPONENT_1,OPPONENT_2,OPPONENT_3,OPPONENT_4,OPPONENT_5,OPPONENT_6,OPPONENT_7,OPPONENT_8,OPPONENT_9,OPPONENT_10,OPPONENT_11,OPPONENT_12,OPPONENT_13,OPPONENT_14,OPPONENT_15,OPPONENT_16,OPPONENT_18,OPPONENT_19,OPPONENT_20,OPPONENT_21,OPPONENT_22,OPPONENT_23,OPPONENT_24,OPPONENT_25,OPPONENT_26,OPPONENT_27,OPPONENT_28,OPPONENT_29,OPPONENT_30,OPPONENT_31,OPPONENT_32,OPPONENT_33,OPPONENT_34,OPPONENT_35,OPPONENT_36')
    new_log.write(new_header)
    for data_line in log[1:]:
        new_data_line = re.sub(r'\(|\)', '', data_line)
        new_log.write(new_data_line)
    log_file.close()
    new_log.close()
