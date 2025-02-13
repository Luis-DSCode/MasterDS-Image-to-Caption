# MasterDS-Image-to-Caption

## Erstellen eines "Personen-Katalogs"

Um die Personenerkennung zu verwenden, muss ein Ordner angelegt werden, in dem Profilbilder/Gesichter aller zu erkennenden Personen abgespeichert sind. Der Dateiname muss der Name der Person sein.

Die Datei **face_download.py** erstellt diesen Ordner automatisch und lädt die Profilbilder herunter. Hierfür ist eine Internetverbindung erforderlich.

Der Ordner kann auch manuell erstellt werden.

## Generieren von Bildbeschreibungen

Die Funktion **generate_image_captions("Bildeingabe", "Profilbilder")** in **Phi_image_to_caption.py** gibt einen String mit der Bildbeschreibung zurück.

- **"Bildeingabe"**: Kann ein einzelnes Bild, ein Ordner mit mehreren Bildern oder ein Link zu einem Bild sein.
- **"Profilbilder"**: Ein Ordner mit Bildern von Personen, die erkannt werden sollen. Der Dateiname muss mit dem Namen der Person übereinstimmen.

Falls ein Bild als **STOCK-BILD** erkannt wird, gibt es momentan nur den String "Stock image" aus.

Beim ersten Aufruf werden die Safetensors des Modells Phi heruntergeladen. Danach kann das Modell lokal und offline verwendet werden.

## Anforderungen (Requirements)

- **Python 3.12.7**
- Abhängigkeiten (siehe `requirements.txt`):

  ```plaintext
  transformers==4.45.2
  Pillow==10.2.0
  torch==2.5.0+cu118
  torchaudio==2.5.0+cu118
  torchvision==0.20.0+cu118
  ExifRead==3.0.0
  requests==2.32.3
  beautifulsoup4==4.12.3
  face-recognition==1.3.0
  numpy==1.26.3

## Verwendung
1. Personen-Katalog erstellen
- Automatisch mit face_download.py
- Manuell: Ordner mit Profilbildern erstellen, wobei der Dateiname der Name der Person ist.

2. Bildbeschreibung generieren:
```
from Phi_image_to_caption import generate_image_captions

# Beispiel: Einzelnes Bild

beschreibung = generate_image_captions("pfad/zum/bild.jpg", "pfad/zum/profilbilder")
print(beschreibung)

# Beispiel: Ordner mit Bildern

beschreibungen = generate_image_captions("pfad/zum/bilder_ordner", "pfad/zum/profilbilder")
for beschreibung in beschreibungen:
    print(beschreibung)
```

## Legacy
Alter Code, andere Modelle und Tests
Nicht wichtig
