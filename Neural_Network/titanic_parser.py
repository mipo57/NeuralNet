import pandas as pd

dataset = pd.read_csv("titanic_dataset.csv", index_col="PassengerId")

dataset["Sex"] = pd.factorize(dataset["Sex"])[0]
dataset["Pclass"] = dataset["Pclass"].fillna(dataset["Pclass"].median())
dataset["Sex"] = dataset["Sex"].fillna(dataset["Sex"].median())
dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
dataset["SibSp"] = dataset["SibSp"].fillna(dataset["SibSp"].median())
dataset["Parch"] = dataset["Parch"].fillna(dataset["Parch"].mean())
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean())
dataset["Embarked"] = dataset["Embarked"].factorize()[0]
dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].median())

Sex_encoded = pd.get_dummies(dataset["Sex"], "Sex")
Parch_encoded = pd.get_dummies(dataset["Parch"], "Parch")
Embarked_encoded = pd.get_dummies(dataset["Embarked"], "Embarked")

important_data = dataset[["Survived", "Pclass", "Age", "SibSp", "Fare"]].join(Sex_encoded).join(Parch_encoded).join(Embarked_encoded)

training = int(0.8 * len(important_data))

#important_data[:3].to_csv("titanic_micro.csv", index=False)
important_data[:training].to_csv("titanic_training.csv", index=False)
important_data[training:].to_csv("titanic_test.csv", index=False)
