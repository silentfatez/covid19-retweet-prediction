"""[Combines and splits the data into files of 100 for easier processing]
"""

df1 = pd.read_table(
    "Data/1.tsv",
    delimiter="\t",
    error_bad_lines=False,
    warn_bad_lines=True,
    header=None,
)
df2 = pd.read_table(
    "Data/2.tsv",
    delimiter="\t",
    error_bad_lines=False,
    warn_bad_lines=True,
    header=None,
)
df3 = pd.read_table(
    "Data/3.tsv",
    delimiter="\t",
    error_bad_lines=False,
    warn_bad_lines=True,
    header=None,
)
dfcomb = df1.append(df2).append(df3).reset_index()

dfcomb.to_feather("clean.ftr")

a = 0
b = 100
for i in range(len(dfcomb)):
    dfcomb[a:b].reset_index().to_feather("split/data_" + str(i) + ".ftr")
    a += 100
    b += 100
