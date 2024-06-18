import glob
import os
import pandas as pd

def main():
    input_dir = 'C:\\Users\\user\\Downloads\\CLT_code-main\\CLT_code-main\\Initial_15,New_10'

    input_files = sorted(glob.glob(os.path.join(input_dir, 'Relocation_ex*.csv')))
    print(f'Found{len(input_files)} files')

    total_relocation = 0

    total_num = len(input_files)

    for input_path in input_files:
        print(f'Processing file : {input_path}')

        input_df = pd.read_csv(input_path)

        relocation = input_df['relocation'].tolist()

        total_relocation += sum(relocation)

    print(f'total_relocation : {total_relocation}')
    print(f'total_num : {total_num}')

    if total_num > 0:
        mean_relocation = total_relocation / total_num
        print(f'mean_relocation : {mean_relocation}')


main()



