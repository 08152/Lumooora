// model.js

let knowledgeBase = [];

// Alle JSON-Dateien automatisch finden
async function loadAllJson() {
  // Hole alle Dateien im Ordner (Browser kann das nicht direkt)
  // → Trick: Wir definieren die Dateiliste manuell
  // Du kannst hier beliebig erweitern:
  const jsonFiles = [
    "daten1.json",
    "daten2.json",
    "faq.json"
  ];

  const allData = [];

  for (const file of jsonFiles) {
    try {
      const res = await fetch(file);
      const data = await res.json();

      if (Array.isArray(data)) {
        allData.push(...data);
      } else {
        allData.push(data);
      }
    } catch (err) {
      console.error("Fehler beim Laden:", file, err);
    }
  }

  knowledgeBase = allData;
  console.log("Geladene Einträge:", knowledgeBase.length);
}

// Mini-Suchmodell
function generateReply(userText) {
  const tokens = userText.toLowerCase().split(/\s+/);

  let bestMatch = null;
  let bestScore = 0;

  for (const entry of knowledgeBase) {
    const frage = (entry.frage || "").toLowerCase();
    let score = 0;

    for (const t of tokens) {
      if (frage.includes(t)) score++;
    }

    if (score > bestScore) {
      bestScore = score;
      bestMatch = entry;
    }
  }

  if (bestMatch) {
    return bestMatch.antwort || "Ich habe etwas gefunden, aber keine Antwort.";
  }

  return "Ich habe nichts Passendes in meinen JSON-Daten gefunden.";
}

// Start
loadAllJson();
