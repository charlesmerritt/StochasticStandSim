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
    def __init__(self,yc,tp,costA,discount,ac):
        """
        Imputs

        """
        #yield curves
        self.yc = yc
        # timber prices
        self.tp = tp
        #cost area
        self.costA = costA
        #annual costs
        self.ac  = ac
        #discount rate
        self.discount = discount
        
        # First we have to define the optimal rotation for the yc
    def optimal_rotation(self):
            
            #Now we merge the age with the yc
            #but keep all in age
            #yc = pd.merge(age,yc, on = "i", how = "left")
            #yc = yc.fillna(0)
            #start with Revenues
            #Clearcuts
            yc = self.yc
            tp = self.tp
            discount = self.discount
            ac = self.ac
            costA = self.costA

            db = pd.merge(yc,tp, on = "pr")
            db["rev_cc"] = (db["cc"]-db["th"])*db["value"]
            db["PV_rev_cc"] = db["rev_cc"]/((1+discount)**db["i"])

            #Thinnings
            db["rev_th"] = db["th"]*db["value"]
            db["PV_rev_th"] = db["rev_th"]/((1+discount)**db["i"])
            #ok now group by age and make index a column
            db = db.groupby(["i","tr"]).sum().reset_index()
            db = db[["i","tr","rev_cc","PV_rev_cc","rev_th","PV_rev_th"]]
            db["Cum_PV_rev_th"] = db["PV_rev_th"].cumsum()
            db["PV_rev"] = db["PV_rev_cc"] + db["Cum_PV_rev_th"]

            ####~~~~Costs~~~~#####
            #Ok first create the age with 0 to age max
            ageMax = max(yc["i"])
            age = pd.DataFrame({"i":range(0,ageMax+1)})

            #Start with area costs
            db1 = pd.merge(age,costA, on = "i", how = "left")
            db1 = db1.fillna(0)
            db1["tr"] = costA["tr"].unique()[0]
            db1["PV_cost"] = db1["cost"]/((1+discount)**db1["i"])
            #ok now annual costs
            db1["ac"]=ac
            db1["PV_ac"] = db1["ac"]/((1+discount)**db1["i"])
            
            db1["PV_cost_F"] = db1["PV_cost"].cumsum() + db1["PV_ac"].cumsum()

            #Ok now merge revenues with costs
            db = pd.merge(db1,db, on = ["i","tr"], how = "left")
            db["PV_rev"] = db["PV_rev"].fillna(0)
            db["NPV"] = db["PV_rev"] - db["PV_cost_F"]
            db["LEV"] = (db["NPV"]*((1+discount)**db["i"]))/(((1+discount)**db["i"])-1)
            db_f = db[db["LEV"]== max(db["LEV"])].reset_index()
            #db_f is the optimal rotation information
            # db for all ages.             
            return [db_f,db]
    
    def forest_value0(self):
            #Ok from the optimal rotation we can calculate the forest values
            db_f = self.optimal_rotation()[0]
            db = self.optimal_rotation()[1]
            discount = self.discount
            #age = optimal_rotation()[1]
            #merge db and age and keep all in age
            opt_age = db_f["i"].values[0]
            LEV = db_f["LEV"].values[0]
            db["FV"] = 0 #Forest Value

            #start with bareland if i ==0 FV  = LEV
            db["Bareland"] = 0
            db["Bareland"][db["i"]==0] = LEV

            #now if i>= age optimal rotation
            db["i_plus"] =0
            db["i_plus"][db["i"]>opt_age] = db["rev_cc"] + db["rev_th"] - db["cost"] - db["ac"] + LEV

            #now if i < age optimal rotation and i> 0
            db["i_minus"] = 0
            db_minus = db[(db["i"]<=opt_age) & (db["i"]>0)]
            for i in db_minus["i"].unique():
                print(i)
                ref= i
                db_minusA = db_minus[db_minus["i"] >= ref]
                db_minusA["i_minus"] = 0
                db_minusA["PV_cost"] = db_minusA["cost"]/((1+discount)**(db_minusA["i"]-ref))
                db_minusA["PV_ac"] = db_minusA["ac"]/((1+discount)**(db_minusA["i"]-ref))
                db_minusA["Cum_cost"] = db_minusA.loc[::-1,"PV_cost"].cumsum()[::-1] + db_minusA.loc[::-1,"PV_ac"].cumsum()[::-1]
                db_minusA["PV_rev_cc"] = db_minusA["rev_cc"][db_minusA["i"] ==opt_age].values[0]/((1+discount)**(opt_age - ref))
                db_minusA["PV_rev_th"] = db_minusA["rev_th"]/((1+discount)**(db_minusA["i"]-ref))
            #na to zero
                db_minusA["PV_rev_th"] = db_minusA["PV_rev_th"].fillna(0)
                db_minusA["PV_rev_th_cum"] = db_minusA.loc[::-1,"PV_rev_th"].cumsum()[::-1]
                db_minusA["NPV"] = db_minusA["PV_rev_cc"] + db_minusA["PV_rev_th_cum"] - db_minusA["Cum_cost"]
                db["i_minus"][db["i"]==ref] = db_minusA["NPV"][db_minusA["i"]==ref].values[0]   +  LEV/(1+discount)**(opt_age-ref)


            db["FV"] = db["Bareland"] + db["i_plus"] + db["i_minus"]
            db = db[["i","tr","FV"]]
            #replace na by -1000
            db["tr"] = db["tr"].fillna(-1000)
            trA = db["tr"].unique()
            trA = trA[trA != -1000][0]
            db["tr"] = db["tr"].replace(-1000,trA)
            return db
        
