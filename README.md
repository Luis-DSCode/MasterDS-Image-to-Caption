# MasterDS-Image-to-Caption

##Erstellen einers "Personen-Katalogs"
Um die Personenerkennung zu verwenden, muss ein Ordner angelegt werden, in dem Profilbilder/Gesichter aller zu erkennenden Personen abgespeichert sind.
Die Namen der Datei muss der Name der Person sein.

Die Datei **face_download.py** erstellt diesen Ordner automatisch.

Kann aber natürlich auch manuel erstellt werden

##Generieren von Beschreibungen
Die Funktion **generate_image_captions("Bildeingabe","Profilbilder")** von Phi-Image-to-caption.py returned einen String mit der Bildbeschreibung 

"Bildeingabe" kann hierbei ein einzelnes Bild, ein Ordner mit mehreren Bildern oder ein Link zu einem Bild sein.

"Profilbilder" sollte ein Ordner mit Bildern von Personen sein, welche erkannt werden sollen, um diese der Beschreibung vorzuhängen. Der Name eines Profilbildes muss mit der Person übereinstimmen.

##Legacy
Legacy Ordner enthält unwichtigen alten Code und zuvor getestete Modelle
