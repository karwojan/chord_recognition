import matplotlib.pyplot as plt
import mlflow

client = mlflow.MlflowClient()

train_loss = client.get_metric_history("74419383bbd640398b4b03f0e7fb4c7c", "train / epoch / loss")
validate_loss = client.get_metric_history("74419383bbd640398b4b03f0e7fb4c7c", "validate / epoch / loss")

train_loss = [x.value for x in train_loss]
validate_loss = [x.value for x in validate_loss]

plt.rcParams.update(
    {
        "font.size": 10,
        "figure.figsize": [12.0, 6.0],
        "figure.autolayout": True,
    }
)
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.plot(train_loss, label="trening")
plt.plot(validate_loss, label="walidacja")
plt.grid()
plt.legend()
plt.savefig("thesis/images/mae2_loss.png")
plt.close()
