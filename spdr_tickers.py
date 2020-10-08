import pandas as pd
import glob


# files = glob.glob("spdr_holdings/*.csv")
#
# df = pd.DataFrame()
#
# for file in files:
#
#     index = file.split("-")[-1].split(".")[0].upper()
#
#     t = pd.read_csv(file, header=1)
#     t['Index'] = index
#     print(t)
#     t.drop("Unnamed: 8", axis=1, inplace=True)
#     df = df.append(t, ignore_index=True)
#
# df.to_csv("spdr_holdings-all.csv")
# print(df)
# # print(df['Symbol'].unique())
#



sectors = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

csv_url = "https://www.sectorspdr.com/sectorspdr/IDCO.Client.Spdrs.Holdings/Export/ExportCsv?symbol="

df = pd.DataFrame()

for tick in sectors:

    t = pd.read_csv(csv_url + tick, header=1)
    t['Index'] = tick
    t.drop("Unnamed: 8", axis=1, inplace=True)
    df = df.append(t, ignore_index=True)

print(df)
