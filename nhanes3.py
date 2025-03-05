import requests
import pandas as pd
import os

class NHANES3Data:
    def __init__(self, file_ext, data_files):
        self.urls = [f"https://wwwn.cdc.gov/nchs/data/nhanes3/{ext}/{file}" for ext in file_ext for file in data_files]
        self.data_files = data_files
        self.out = pd.DataFrame(columns=['SEQN'])

    def download_files(self):
        for url in self.urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    filename = url.split('/')[-1]
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {filename}")
                else:
                    print(f"Failed to download: {url}")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")

    def merge_files(self):
        for file in self.data_files:
            tmp = pd.read_sas(file)
            self.out = pd.merge(self.out, tmp, on='SEQN', how='outer')

    def clean_directory(self):
        for file in os.listdir():
            if file.endswith(".xpt"):
                os.remove(file)

    def get_nhanes3(self):
        print("Downloading SAS files...")
        self.download_files()
        print("Merging files...")
        self.merge_files()
        print("Cleaning directory...")
        self.clean_directory()
        return self.out

# Call function to get data
dat = NHANES3Data(file_ext=['34a', '34B'], data_files=['HGUHS.xpt', 'HGUHSSE.xpt']).get_nhanes3()

# Select columns of interest
dat = dat[['SEQN', 'GUPHSPF', 'GURHSPF']]

# Update "HS primary finding" based on the given instructions
dat['HS primary finding'] = dat.apply(
    lambda row: row['GURHSPF'] if pd.notna(row['GURHSPF']) else row['GUPHSPF'],
    axis=1
)

dat['HS primary finding'].value_counts()

# Load other datasets
labs = pd.read_csv('data/lab_data.csv')
exam = pd.read_csv('data/exam_data.csv')
adult = pd.read_csv('data/adult_data.csv')

adult = adult[adult['Age at interview (screener) - qty'] >= 18]

labs = labs[(labs['Serum hepatitis B core antibody'] == 2) | (labs['Serum hepatitis B core antibody'] == 8) | (labs['Serum hepatitis B core antibody'].isna())]

exam = exam[exam['Age at interview (Screener)'] >= 18]

adult = adult[(adult['Have you consumed alcohol in last 30 min'] == 2) | (adult['Have you consumed alcohol in last 30 min'] == 8) | (adult['Have you consumed alcohol in last 30 min'].isna())]

# Rename the first column in adult, labs, and exam to SEQN
adult.rename(columns={adult.columns[0]: 'SEQN'}, inplace=True)
labs.rename(columns={labs.columns[0]: 'SEQN'}, inplace=True)
exam.rename(columns={exam.columns[0]: 'SEQN'}, inplace=True)

# Remove duplicate columns from labs and exam that are already in adult
labs = labs[[col for col in labs.columns if col not in adult.columns or col == 'SEQN']]
exam = exam[[col for col in exam.columns if col not in adult.columns or col == 'SEQN']]

# Inner join datasets together by SEQN 
dat = dat.merge(adult, on='SEQN', how='inner')
dat = dat.merge(labs, on='SEQN', how='left')
dat = dat.merge(exam, on='SEQN', how='left')

# Filter based on "HS primary finding" and "GUPHSPF"
dat = dat[(dat['HS primary finding'].isin([2, 3, 4]))]

dat.to_csv('nhanes3.csv',index=False)