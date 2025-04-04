from comet import download_model, load_from_checkpoint

# 🔻 Download & load model
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

# 🧪 Test Data
data = [
    {
        "src": "The output signal provides constant sync so the display never glitches.",
        "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
    },
    {
        "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
        "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
    },
    {
        "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
        "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
    }
]

# 🚀 Run Prediction
model_output = model.predict(data, batch_size=8, gpus=1)

# 📤 Print Result
print(model_output)
