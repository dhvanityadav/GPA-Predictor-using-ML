import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from matplotlib import pyplot as plt

# import pickle
# # Save model to file
# with open("gpa_model.pkl", "wb") as f:
#     pickle.dump(model, f)


data = pd.read_csv('Student_performance_data _.csv')

print(data.isnull().sum())

data["StudyTimeWeeklyInt"] = data["StudyTimeWeekly"].round()

# input and output variables
X = data[["StudyTimeWeeklyInt", "Absences", "Tutoring"]]
y = data["GPA"]

# train the model

model = LinearRegression()
model.fit(X, y)
predicted_GPA = model.predict(X)

# valid regression metrics

mae = mean_absolute_error(y, predicted_GPA)
mse = mean_squared_error(y, predicted_GPA)
rmse = np.sqrt(mse)
r2 = r2_score(y, predicted_GPA)
print(f"MAE: {round(mae, 2)}, \nMSE: {round(mse, 2)}, \nRMSE: {round(rmse, 2)}, \nR2: {round(r2, 4)}")
# r2_score is closer to 1, better the model


# scatter plot + Regression line
plt.figure(figsize=(10, 6))
plt.scatter(data["StudyTimeWeeklyInt"], y, color='skyblue', label="Actual GPA")
plt.plot(data["StudyTimeWeeklyInt"], predicted_GPA, color="red", label="Predicted GPA (Regression Line)")
plt.title("Model prediction VS Actual GPA")
plt.xlabel("Study Time Weekly Int")
plt.ylabel("Final Output GPA")
plt.grid(True)
plt.show()


new_study_hours = int(input("Enter the number of hours you study weekly: "))
new_absences = int(input("Enter the number of absences you have: "))
new_tutoring = input("Enter Yes for tutoring & No for not: ").strip().lower()
if new_tutoring == "yes":
    new_tutoring = 1
elif new_tutoring == "no":
    new_tutoring = 0


new_data = pd.DataFrame([[new_study_hours, new_absences, new_tutoring]], columns=["StudyTimeWeeklyInt", "Absences", "Tutoring"])
new_prediction = model.predict(new_data)


if new_tutoring == 1:
    new_tutoring = "We take tutoring weekly"
elif new_tutoring == 0:
    new_tutoring = "No we don't take tutoring weekly"

print(f"Predicted GPA for {new_study_hours} hours of study weekly, {new_tutoring}, and {new_absences} absences is: {new_prediction[0].round(2)}")

