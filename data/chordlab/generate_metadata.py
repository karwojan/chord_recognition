import pandas as pd
from glob import glob

rs200_original = pd.read_csv("./rs200/rs200.txt", delimiter="\t", header=None, names=["filepath", "song_rank", "song", "artist", "year", "original"])
rs200 = pd.DataFrame()
rs200["filepath"] = "./rs200/" + rs200_original["filepath"] + ".lab"
rs200["song"] = rs200_original["song"]
rs200["artist"] = rs200_original["artist"]
rs200["year"] = rs200_original["year"]
rs200["subset"] = "rs200"
rs200 = rs200.dropna(axis="index")

rwc_pop_original = pd.read_csv("./rwc_pop/rwc_pop.txt", delimiter="\t")
rwc_pop = pd.DataFrame()
rwc_pop["filepath"] = rwc_pop_original.apply(lambda x: f"./rwc_pop/N{int(x.iloc[0].split(' ')[1]):03}-{x.iloc[1]}-T{x.iloc[2].split(' ')[1]}.lab", axis=1)
rwc_pop["song"] = rwc_pop_original["Title"]
rwc_pop["artist"] = rwc_pop_original["Artist (Vocal)"]
rwc_pop["subset"] = "rwc_pop"
rwc_pop = rwc_pop.dropna(axis="index")

isophonics = pd.DataFrame()
isophonics["filepath"] = glob("./isophonics/**/*.lab", recursive=True)
isophonics["song"] = isophonics["filepath"].str.split("/", expand=True)[4]
isophonics["song"] = isophonics["song"].str.replace(r"^.*_-_|^\d+ ","",regex=True)
isophonics["song"] = isophonics["song"].str.replace(r"_"," ",regex=True)
isophonics["song"] = isophonics["song"].str.replace(r"\.lab","",regex=True)
isophonics["album"] = isophonics["filepath"].str.split("/", expand=True)[3]
isophonics["album"] = isophonics["album"].str.replace(r".*_-_|\d+ ","",regex=True)
isophonics["album"] = isophonics["album"].str.replace(r"_"," ",regex=True)
isophonics["artist"] = isophonics["filepath"].str.split("/", expand=True)[2]
isophonics["subset"] = "isophonics"
isophonics = isophonics.dropna(axis="index")

billboard_ids = {int(filename.split("/")[2]) for filename in glob("./mcgill_billboard/**/*.lab", recursive=True)}
billboard_original = pd.read_csv("./mcgill_billboard/billboard-2.0-index.csv").filter(billboard_ids, axis="index")
billboard = pd.DataFrame()
billboard["filepath"] = billboard_original.apply(lambda x: f"./mcgill_billboard/{int(x['id']):04}/full.lab", axis=1)
billboard["song"] = billboard_original["title"]
billboard["artist"] = billboard_original["artist"]
billboard["subset"] = "billboard"
billboard = billboard.dropna(axis="index")

uspop = pd.DataFrame()
uspop["filepath"] = glob("./uspop/**/*.lab", recursive=True)
uspop["song"] = uspop["filepath"].str.split("/", expand=True)[4]
uspop["song"] = uspop["song"].str.replace(r"^\d+-", "", regex=True)
uspop["song"] = uspop["song"].str.replace(r"_", " ", regex=True)
uspop["song"] = uspop["song"].str.replace(r"\.lab", "", regex=True)
uspop["album"] = uspop["filepath"].str.split("/", expand=True)[3]
uspop["album"] = uspop["album"].str.replace(r"_", " ", regex=True)
uspop["artist"] = uspop["filepath"].str.split("/", expand=True)[2]
uspop["artist"] = uspop["artist"].str.replace(r"_", " ", regex=True)
uspop["subset"] = "uspop"

compound = pd.concat([rs200, rwc_pop, isophonics, billboard, uspop])
compound.to_csv("./metadata.csv", sep=";", header=True, index=False)
