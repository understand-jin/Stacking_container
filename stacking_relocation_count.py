import glob
import os
import pandas as pd

def main():
    input_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\experiment\\Output_Data_25\\Heiristic_1\\Initial_0\\New_25'

    input_files = sorted(glob.glob(os.path.join(input_dir, 'Configuration_*.csv')))

    total_relocation = 0
    total_num = len(input_files)

    for i in range(len(input_files)):
        input_path = input_files[i]

        input_df = pd.read_csv(input_path)

        relocation = input_df['reloc'].tolist()

        num = input_df['weight'].tolist()

        total_relocation = total_relocation + sum(relocation)

    
    print(f'total_relocation : {total_relocation}')
    print(f'total_num : {total_num}')

    if total_relocation != 0:
       mean_relocation = total_relocation / total_num
       print(f'mean_relocation : {mean_relocation}')
    
main()



