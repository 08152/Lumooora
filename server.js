const express = require("express");
const path = require("path");

const {
    normalize,
    tokenize,
    words,
    containsWord
} = require("./tokenizer");

const app = express();
const PORT = process.env.PORT || 10000;

app.use(express.json({ limit: "2mb" }));

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "index.html"));
});

/* =========================
   EIGENES WISSEN
========================= */

const knowledge = {
    javascript: {
        keywords: ["javascript", "js", "node", "nodejs"],
        facts: [
            "JavaScript ist eine Programmiersprache.",
            "JavaScript kann im Browser und auf Servern laufen.",
            "Mit Node.js kann JavaScript für Server verwendet werden."
        ]
    },

    tokenizer: {
        keywords: ["tokenizer", "token", "tokenisierung"],
        facts: [
            "Ein Tokenizer zerlegt Text in einzelne Bestandteile.",
            "Diese Bestandteile können Wörter, Zahlen und Satzzeichen sein.",
            "Lumora verwendet dafür einen eigenen JavaScript-Tokenizer."
        ]
    },

    render: {
        keywords: ["render", "hosting", "server"],
        facts: [
            "Render kann Node.js-Webserver hosten.",
            "Lumora kann als Node.js-Webservice auf Render laufen."
        ]
    },

    ki: {
        keywords: ["ki", "künstliche", "intelligenz", "ai", "modell"],
        facts: [
            "Eine KI kann Texte analysieren und daraus Ausgaben erzeugen.",
            "Größere Sprachmodelle verwenden neuronale Netze und Transformer.",
            "Lumora kann später um ein eigenes Sprachmodell erweitert werden."
        ]
    }
};

/* =========================
   FRAGE ERKENNEN
========================= */

function detectQuestion(text) {
    const t = normalize(text);

    if (
        t.includes("was ist") ||
        t.includes("was bedeutet") ||
        t.includes("erklär") ||
        t.includes("erklaer") ||
        t.includes("erkläre") ||
        t.includes("wie funktioniert")
    ) {
        return "explain";
    }

    if (
        t.includes("wer bist du") ||
        t.includes("was bist du") ||
        t.includes("wie heißt du") ||
        t.includes("wie heisst du")
    ) {
        return "identity";
    }

    if (
        t.includes("kannst du") ||
        t.includes("kannst")
    ) {
        return "ability";
    }

    if (
        t.endsWith("?") ||
        t.startsWith("warum") ||
        t.startsWith("wie")
    ) {
        return "question";
    }

    return "statement";
}

/* =========================
   THEMA ERKENNEN
========================= */

function detectTopic(text) {
    const tokenList = words(text);

    let bestTopic = null;
    let bestScore = 0;

    for (const [topic, data] of Object.entries(knowledge)) {
        let score = 0;

        for (const keyword of data.keywords) {
            if (tokenList.includes(normalize(keyword))) {
                score++;
            }
        }

        if (score > bestScore) {
            bestScore = score;
            bestTopic = topic;
        }
    }

    return bestTopic;
}

/* =========================
   EIGENE ANTWORT ERZEUGEN
========================= */

function generateOwnAnswer(message) {
    const questionType = detectQuestion(message);
    const topic = detectTopic(message);

    if (questionType === "identity") {
        return [
            "Ich bin Lumora.",
            "Ich bin eine selbst entwickelte JavaScript-KI.",
            "Mein Text wird zuerst mit tokenizer.js zerlegt.",
            "Danach analysiere ich die Frage und suche nach passendem Wissen."
        ].join(" ");
    }

    if (
        containsWord(message, "hallo") ||
        containsWord(message, "hi") ||
        containsWord(message, "hey") ||
        containsWord(message, "servus")
    ) {
        const greetings = [
            "Hallo! Ich bin Lumora.",
            "Hey! Lumora ist bereit.",
            "Hallo! Ich habe deine Nachricht analysiert."
        ];

        return greetings[
            Math.floor(Math.random() * greetings.length)
        ];
    }

    if (
        containsWord(message, "danke") ||
        containsWord(message, "dankeschön") ||
        containsWord(message, "dankeschoen")
    ) {
        return "Gerne!";
    }

    if (topic) {
        const data = knowledge[topic];

        if (questionType === "explain") {
            const intro = {
                javascript: "JavaScript ist eine Programmiersprache.",
                tokenizer: "Ein Tokenizer ist ein Werkzeug zur Textanalyse.",
                render: "Render ist eine Plattform für Webservices.",
                ki: "Künstliche Intelligenz bezeichnet Systeme, die Daten analysieren und daraus Ausgaben erzeugen."
            };

            const facts = [...data.facts];

            facts.sort(() => Math.random() - 0.5);

            return (
                intro[topic] +
                " " +
                facts
                    .slice(0, Math.min(2, facts.length))
                    .join(" ")
            );
        }

        if (questionType === "ability") {
            return (
                "Ja. " +
                data.facts[0] +
                " " +
                data.facts[1]
            );
        }

        return (
            "Ich habe Informationen über " +
            topic +
            ". " +
            data.facts[
                Math.floor(Math.random() * data.facts.length)
            ]
        );
    }

    const tokenList = tokenize(message);
    const wordList = words(message);

    if (wordList.length === 0) {
        return "Schreib mir etwas, damit ich es analysieren kann.";
    }

    if (questionType === "question") {
        return (
            "Ich habe deine Frage analysiert. " +
            "Sie enthält " +
            wordList.length +
            " Wörter und " +
            tokenList.length +
            " Tokens. " +
            "Dazu habe ich momentan noch kein passendes Wissen."
        );
    }

    return (
        "Ich habe deinen Text analysiert. " +
        "Er enthält " +
        wordList.length +
        " Wörter und " +
        tokenList.length +
        " Tokens. " +
        "Mein Wissen kann später erweitert werden."
    );
}

/* =========================
   CHAT API
========================= */

app.post("/api/chat", (req, res) => {
    try {
        const message = String(req.body?.message || "").trim();

        if (!message) {
            return res.status(400).json({
                error: "Keine Nachricht."
            });
        }

        const tokens = tokenize(message);
        const questionType = detectQuestion(message);
        const topic = detectTopic(message);
        const answer = generateOwnAnswer(message);

        res.json({
            answer,
            tokens,
            questionType,
            topic
        });
    } catch (error) {
        console.error(error);

        res.status(500).json({
            error: "Interner KI-Fehler."
        });
    }
});

/* =========================
   TOKENIZER API
========================= */

app.post("/api/tokenize", (req, res) => {
    const text = String(req.body?.text || "");

    res.json({
        text,
        tokens: tokenize(text)
    });
});

/* =========================
   HEALTH
========================= */

app.get("/api/health", (req, res) => {
    res.json({
        online: true,
        ai: "Lumora",
        tokenizer: true,
        ownAnswerGenerator: true,
        topics: Object.keys(knowledge)
    });
});

/* =========================
   SERVER
========================= */

app.listen(PORT, "0.0.0.0", () => {
    console.log("================================");
    console.log("          LUMORA ONLINE");
    console.log("================================");
    console.log("Tokenizer: AKTIV");
    console.log("Antwortgenerator: AKTIV");
    console.log("Wissensdatenbank: AKTIV");
    console.log("Port:", PORT);
});
