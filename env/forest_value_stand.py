# -*- coding: utf-8 -*-
"""
Stochastic Harvest Schedule
#First stage -> plant
#Second stage -> harvest with and without fertilization

#author: bdasilva
1/30/2024
"""
import pandas as pd

class forest_value():
    def __init__(self, yc, tp, costA, discount, ac):
        """
        Inputs:
        - yc: yield curves DataFrame
        - tp: timber prices DataFrame
        - costA: cost area DataFrame
        - discount: discount rate (float)
        - ac: annual cost (float)
        """
        self.yc = yc
        self.tp = tp
        self.costA = costA
        self.ac = ac
        self.discount = discount

    def optimal_rotation(self):
        # Merge yield curves with timber prices
        db = pd.merge(self.yc, self.tp, on="pr")
        db["rev_cc"] = (db["cc"] - db["th"]) * db["value"]
        db["PV_rev_cc"] = db["rev_cc"] / ((1 + self.discount) ** db["i"])

        # Thinnings
        db["rev_th"] = db["th"] * db["value"]
        db["PV_rev_th"] = db["rev_th"] / ((1 + self.discount) ** db["i"])

        # Group by age and treatment
        db = db.groupby(["i", "tr"]).sum().reset_index()
        db["Cum_PV_rev_th"] = db.groupby("tr")["PV_rev_th"].cumsum()
        db["PV_rev"] = db["PV_rev_cc"] + db["Cum_PV_rev_th"]

        # Cost calculations
        ageMax = max(self.yc["i"])
        age = pd.DataFrame({"i": range(0, ageMax + 1)})

        db1 = pd.merge(age, self.costA, on="i", how="left").fillna(0)
        db1.loc[:, "tr"] = self.costA["tr"].unique()[0]
        db1.loc[:, "PV_cost"] = db1["cost"] / ((1 + self.discount) ** db1["i"])
        db1.loc[:, "ac"] = self.ac
        db1.loc[:, "PV_ac"] = db1["ac"] / ((1 + self.discount) ** db1["i"])
        db1["PV_cost_F"] = db1["PV_cost"].cumsum() + db1["PV_ac"].cumsum()

        # Merge revenues with costs
        db = pd.merge(db1, db, on=["i", "tr"], how="left").fillna(0)
        db["NPV"] = db["PV_rev"] - db["PV_cost_F"]
        db["LEV"] = (db["NPV"] * ((1 + self.discount) ** db["i"])) / (((1 + self.discount) ** db["i"]) - 1)

        # Find optimal rotation
        if db["LEV"].max() == -float("inf"):
            raise ValueError("No positive LEV found. Check your inputs.")

        db_f = db[db["LEV"] == db["LEV"].max()].reset_index(drop=True)

        # Ensure complete age range
        full_age_range = pd.DataFrame({"i": range(0, ageMax + 1)})
        db = pd.merge(full_age_range, db, on="i", how="left").fillna(0)

        return [db_f, db]

    def forest_value0(self):
        db_f, db = self.optimal_rotation()
        opt_age = db_f.loc[0, "i"]
        LEV = db_f.loc[0, "LEV"]

        # Initialize FV and Bareland
        db["FV"] = 0
        db["Bareland"] = 0
        db.loc[db["i"] == 0, "Bareland"] = LEV

        # Post-optimal rotation (i >= opt_age)
        db["i_plus"] = 0
        db.loc[db["i"] >= opt_age, "i_plus"] = (
            db["rev_cc"] + db["rev_th"] - db["cost"] - db["ac"] + LEV
        )

        # Pre-optimal rotation (0 < i < opt_age)
        db["i_minus"] = 0
        db_minus = db[(db["i"] <= opt_age) & (db["i"] > 0)].copy()

        for ref in db_minus["i"].unique():
            db_minusA = db_minus[db_minus["i"] >= ref].copy()
            db_minusA.loc[:, "PV_cost"] = db_minusA["cost"] / ((1 + self.discount) ** (db_minusA["i"] - ref))
            db_minusA.loc[:, "PV_ac"] = db_minusA["ac"] / ((1 + self.discount) ** (db_minusA["i"] - ref))
            db_minusA.loc[:, "Cum_cost"] = (
                db_minusA.loc[::-1, "PV_cost"].cumsum()[::-1] +
                db_minusA.loc[::-1, "PV_ac"].cumsum()[::-1]
            )

            if (db_minusA["i"] == opt_age).sum() == 0:
                print(f"Warning: Optimal age {opt_age} not found in db_minusA for ref {ref}")
                continue

            db_minusA.loc[:, "PV_rev_cc"] = (
                db_minusA.loc[db_minusA["i"] == opt_age, "rev_cc"].values[0] /
                ((1 + self.discount) ** (opt_age - ref))
            )
            db_minusA.loc[:, "PV_rev_th"] = db_minusA["rev_th"] / ((1 + self.discount) ** (db_minusA["i"] - ref))
            db_minusA.loc[:, "PV_rev_th"] = db_minusA["PV_rev_th"].fillna(0)
            db_minusA.loc[:, "PV_rev_th_cum"] = db_minusA.loc[::-1, "PV_rev_th"].cumsum()[::-1]
            db_minusA.loc[:, "NPV"] = db_minusA["PV_rev_cc"] + db_minusA["PV_rev_th_cum"] - db_minusA["Cum_cost"]

            if db_minusA[db_minusA["i"] == ref].empty:
                raise ValueError(f"No matching ref age {ref} in db_minusA. Check your inputs.")

            db.loc[db["i"] == ref, "i_minus"] = (
                db_minusA.loc[db_minusA["i"] == ref, "NPV"].values[0] +
                LEV / ((1 + self.discount) ** (opt_age - ref))
            )

        # Final Forest Value
        db["FV"] = db["Bareland"] + db["i_plus"] + db["i_minus"]
        db = db[["i", "tr", "FV"]]

        # Replace missing treatment values
        db["tr"] = db["tr"].fillna(-1000)
        trA = db["tr"].unique()
        trA = trA[trA != -1000][0] if len(trA) > 1 else -1000
        db["tr"] = db["tr"].replace(-1000, trA)

        return db
