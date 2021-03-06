import pandas as pd
from glob import glob

# source - rs200
rs200_original = pd.read_csv("./data/chordlab/rs200/rs200.txt", delimiter="\t", header=None, names=["filepath", "song_rank", "song", "artist", "year", "original"])
rs200 = pd.DataFrame()
rs200["filepath"] = "./data/chordlab/rs200/" + rs200_original["filepath"] + "_dt.clt"
rs200["song"] = rs200_original["song"]
rs200["artist"] = rs200_original["artist"]
rs200["subset"] = "rs200"
rs200 = rs200.dropna(axis="index")

# source - rwc pop
rwc_pop_original = pd.read_csv("./data/chordlab/rwc_pop/rwc_pop.txt", delimiter="\t")
rwc_pop = pd.DataFrame()
rwc_pop["filepath"] = rwc_pop_original.apply(lambda x: f"./data/chordlab/rwc_pop/N{int(x.iloc[0].split(' ')[1]):03}-{x.iloc[1]}-T{x.iloc[2].split(' ')[1]}.lab", axis=1)
rwc_pop["song"] = rwc_pop_original["Title"]
rwc_pop["artist"] = rwc_pop_original["Artist (Vocal)"]
rwc_pop["subset"] = "rwc_pop"
rwc_pop = rwc_pop.dropna(axis="index")

# source - isophonics
isophonics = pd.DataFrame()
isophonics["filepath"] = glob("./data/chordlab/isophonics/**/*.lab", recursive=True)
isophonics["song"] = isophonics["filepath"].str.split("/", expand=True)[6]
isophonics["song"] = isophonics["song"].str.replace(r"^.*_-_|^\d+ ","",regex=True)
isophonics["song"] = isophonics["song"].str.replace(r"_"," ",regex=True)
isophonics["song"] = isophonics["song"].str.replace(r"\.lab","",regex=True)
isophonics["album"] = isophonics["filepath"].str.split("/", expand=True)[5]
isophonics["album"] = isophonics["album"].str.replace(r".*_-_|\d+ ","",regex=True)
isophonics["album"] = isophonics["album"].str.replace(r"_"," ",regex=True)
isophonics["artist"] = isophonics["filepath"].str.split("/", expand=True)[4]
isophonics["subset"] = "isophonics"
isophonics = isophonics.dropna(axis="index")

# source - billboard
billboard_ids = {int(filename.split("/")[4]) for filename in glob("./data/chordlab/mcgill_billboard/**/*.lab", recursive=True)}
billboard_original = pd.read_csv("./data/chordlab/mcgill_billboard/billboard-2.0-index.csv").filter(billboard_ids, axis="index")
billboard = pd.DataFrame()
billboard["filepath"] = billboard_original.apply(lambda x: f"./data/chordlab/mcgill_billboard/{int(x['id']):04}/full.lab", axis=1)
billboard["song"] = billboard_original["title"]
billboard["artist"] = billboard_original["artist"]
billboard["subset"] = "billboard"
billboard = billboard.dropna(axis="index")

# source - uspop
uspop = pd.DataFrame()
uspop["filepath"] = glob("./data/chordlab/uspop/**/*.lab", recursive=True)
uspop["song"] = uspop["filepath"].str.split("/", expand=True)[6]
uspop["song"] = uspop["song"].str.replace(r"^\d+-", "", regex=True)
uspop["song"] = uspop["song"].str.replace(r"_", " ", regex=True)
uspop["song"] = uspop["song"].str.replace(r"\.lab", "", regex=True)
uspop["album"] = uspop["filepath"].str.split("/", expand=True)[5]
uspop["album"] = uspop["album"].str.replace(r"_", " ", regex=True)
uspop["artist"] = uspop["filepath"].str.split("/", expand=True)[4]
uspop["artist"] = uspop["artist"].str.replace(r"_", " ", regex=True)
uspop["subset"] = "uspop"

# source - robbie williams
robbie = pd.DataFrame()
robbie["filepath"] = glob("./data/chordlab/robbie_williams/**/*.txt", recursive=True)
robbie["song"] = robbie["filepath"].str.split("/", expand=True)[5]
robbie["song"] = robbie["song"].str.replace(r"^([0123456789abc]+-)+", "", regex=True)
robbie["song"] = robbie["song"].str.replace(r"GTChords.txt", "", regex=False)
robbie["album"] = robbie["filepath"].str.split("/", expand=True)[4]
robbie["album"] = robbie["album"].str.replace(r"\d{4}-", "", regex=True)
robbie["artist"] = "Robbie Williams"
robbie["subset"] = "robbie_williams"

# concatenate to single index
compound = pd.concat([rs200, rwc_pop, isophonics, billboard, uspop, robbie]).reset_index(drop=True)

# filter duplicates
compound["_song"] = compound["song"].str.lower().str.strip()
compound["_artist"] = compound["artist"].str.lower().str.strip()
compound["_album"] = compound["album"].str.lower().str.strip()
for song_and_artist, probably_same_songs in compound.groupby(["_song", "_artist"]):
    for album, really_same_songs in probably_same_songs.groupby("album", dropna=False):
        if not isinstance(album, str) and not probably_same_songs["album"].isna().all():
            compound.drop(really_same_songs.index, inplace=True)
        else:
            compound.drop(really_same_songs.iloc[1:].index, inplace=True)
compound.drop(["_song", "_artist", "_album"], axis=1, inplace=True)

# save to file
compound.to_csv("./data/index.csv", sep=";", header=True, index=False)
