import pandas as pd
import scanpy as sc
import os

def getPairs(path, genes_use, species):
    db = pd.read_csv(path)
    db = db.loc[db["species"] == species, ]
    db = db.loc[(db.iloc[:, 0].isin(genes_use))&(db.iloc[:, 1].isin(genes_use)), ]
    return db


def getInteractionDB(genes_use, species, lr_path = "./prior_db/lr_pair.csv",
                     pathways_path = "./prior_db/pathways.csv",
                     tftg_path = "./prior_db/TFTG.csv",
                     rec_tf_path = "./prior_db/rec_tf.csv"):
    pathways = getPairs(pathways_path, genes_use, species)
    rec_tf = getPairs(rec_tf_path, genes_use, species)
    tftg = getPairs(tftg_path, genes_use, species)
    tftg = tftg.loc[tftg["src"].isin(rec_tf["tf"].unique()),] 
    pathways = pd.concat([pathways, tftg], ignore_index=True).drop_duplicates(keep=False)
    
    rec_tftg = rec_tf.merge(tftg,how="inner", left_on='tf', right_on='src')
    rec_tftg = rec_tftg.iloc[:, [0,1,4,2]]
    rec_tftg.columns = rec_tftg.columns.str.replace("species_x", "species")
    rec_tg = rec_tftg[["receptor", "dest", "species"]].drop_duplicates()
    rec_tg = rec_tg.loc[rec_tg.iloc[:, 0] != rec_tg.iloc[:, 1],:]
    receptor = rec_tf["receptor"].unique()
    lr_db = getPairs(lr_path, genes_use, species)
    lr_db = lr_db.loc[lr_db["receptor"].isin(receptor),].reset_index(drop=True)
    
    
    return lr_db, pathways, rec_tg, tftg
    
