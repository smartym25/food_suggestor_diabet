import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def cgm_analisys_graphic(id_sub, start_date, end_date):
    
    cgm = pd.read_csv(f"csv-files\subjects-data\Subject_{id_sub}.csv")
    
    cgm['EventDateTime'] = pd.to_datetime(cgm['EventDateTime'])

    date_range = pd.date_range(start=start_date, end=end_date)

    date_list = [d.strftime("%Y-%m-%d") for d in date_range]

    cgm_food = cgm[(cgm["TotalBolusInsulinDelivered"] != 0 ) & (cgm["FoodDelivered"] != 0)]

    cgm_insulyn = cgm[(cgm["TotalBolusInsulinDelivered"] != 0)  & (cgm["FoodDelivered"] == 0)]

    for i in range(0,len(date_list)): 
    #i viene utilizzato solo con le liste con i numeri, se voglio utilizzare delle stringe devo definire prima una variabile

        current_day = date_list[i]
        
        cgm_filtered2 = cgm[cgm['EventDateTime'].dt.date == pd.to_datetime(current_day).date()]

        cgm_food_2 = cgm_food[cgm_food['EventDateTime'].dt.date == pd.to_datetime(current_day).date()]

        cgm_insulyn_2 = cgm_insulyn[cgm_insulyn['EventDateTime'].dt.date == pd.to_datetime(current_day).date()]

        fig, ax = plt.subplots(figsize=(20, 10))
        
        sns.lineplot(data=cgm_filtered2, x="EventDateTime", y="CGM", color="lightblue", ax=ax)

        sns.scatterplot(data=cgm_food_2, x="EventDateTime", y="CGM", color="orange", size="TotalBolusInsulinDelivered", 
                        edgecolor="grey", linewidth=1, ax=ax)
    
        sns.scatterplot(data=cgm_insulyn_2, x="EventDateTime", y="CGM", color="green", size="TotalBolusInsulinDelivered", 
                        edgecolor="grey", linewidth=1, ax=ax)
        
        plt.axhline(y=180, color='red', linestyle='--', linewidth=2) #hyperglycemic zone
        
        plt.axhline(y=250, color='darkred', linestyle='-', linewidth=2.5) #danger
        
        plt.axhline(y=70, color='blue', linestyle='--', linewidth=2) #hypoglycemic zone
        
        plt.axhline(y=54, color='darkblue', linestyle='-', linewidth=2.5) #danger
        
        plt.yticks(np.arange(25, 300, 25))
        
        plt.show()

cgm_analisys_graphic(8,"2024-01-16","2024-01-16")

class predict_suggestor_food():
    def __init__(self, id_num):
        self.id_num = id_num
        self.sub_csv = pd.read_csv(f"csv-files/subjects-data/Subject_{self.id_num}.csv")
        self.data_subj = pd.read_csv("indicators_subjects.csv", encoding='latin1')
        self.data_food = pd.read_csv("csv-files/food-data/food_data.csv", encoding='latin1')
        self.activity_factor = self.calculate_activity_factor()
        
    def calculate_activity_factor(self):
        exercise = self.sub_csv[self.sub_csv["DeviceMode"] == "exercise"]

        num_exercise = exercise.shape[0]
        num_total =  self.sub_csv.shape[0]

        percentage = (num_exercise / num_total) * 100

        if 90 <= percentage <= 100:
            activity_factor = 1.9

        elif 70 <= percentage <= 80:
            activity_factor = 1.725

        elif 50 <= percentage <= 60:
            activity_factor = 1.55

        elif 30 <= percentage <= 40:
            activity_factor = 1.375

        else:
            activity_factor = 1.2
        return activity_factor
    
    def tdee(self): #fabbisogno calorico giornaliero espresso in kcal
        id_data_subj = self.data_subj[self.data_subj["ID"] == self.id_num]

        weight = id_data_subj["Weight_kg"].values[0]
        height = id_data_subj["Height_cm"].values[0]
        sex = id_data_subj["Sex"].values[0]
        age = id_data_subj["Age"].values[0]
        
        if sex == "Male":
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

        tdee = bmr * self.activity_factor
        return round(tdee, 1)
    
    def consume_carb_pday(self): #consumo giornaliero di carboidrati in grammi al giorno in media 
        self.sub_csv['EventDateTime'] = pd.to_datetime(self.sub_csv['EventDateTime'])
        
        start_date = "2024-01-12"
        end_date = "2024-02-08"

        food_pday = []

        for single_date in pd.date_range(start=start_date, end=end_date, freq='D'):

            mask = (self.sub_csv['EventDateTime'].dt.date == pd.to_datetime(single_date).date())
            
            df = self.sub_csv[mask]
            carb_pday = df[df["CarbSize"] != 0]["CarbSize"].sum()

            food_pday.append(carb_pday)

        medium_carb_pday = sum(food_pday) / len(food_pday)

        calories_carb_pday = medium_carb_pday * 4  #(kcal)
        return calories_carb_pday
    
    def calculate_cg(self):
        igs = self.data_food["Ig"]
        carbs = self.data_food["Cho"]

        cgs = []
        for i in range(len(igs)):
            cg = (igs[i] * carbs[i]) / 100 
            cgs.append(cg)

        self.data_food["Cg"] = cgs
        return cgs

    def class_ig(self):
        igs = self.data_food["Ig"]

        ig_class = []
        for ig in igs:
            if ig <= 55:
                ig_class.append("low")
            elif 56 <= ig <= 69:
                ig_class.append("medium")
            else:
                ig_class.append("high")

        self.data_food["CategoryIg"] = ig_class
        return ig_class

#costruire il modello che predice attività glicemica 10 giorni dopo 
#consigliare una dieta per ogni 10 giorni 
# zone di iperglicemia --> cibi a basso indice glicemico e cibi che rallentano glucosio (carne e pesce)
# zone di ipoglicemia --> ig alto e ig medio per mantenere stabilità e insulina media unità
# 6-10 colazione, 11-12 spuntino, 13-14:30 pranzo, 16-19 spuntino, 20-21 cena ad ogni intervallo corrisponde un food intake 
# quando modalità exercise dopo controllare che cgm non si abbassi troppo velocmente nel caso cibo 
# 3 pranzi principales eguire scehema usl cell 