```js
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


/* =========================================
   EIGENES WISSEN
========================================= */

const knowledge = {
    javascript: {
        keywords: [
            "javascript",
            "js",
            "node",
            "nodejs"
        ],

        facts: [
            "JavaScript ist eine Programmiersprache.",
            "JavaScript kann im Browser und auf Servern laufen.",
            "Mit Node.js kann JavaScript für Server verwendet werden."
        ]
    },

    tokenizer: {
        keywords: [
            "tokenizer",
            "token",
            "tokenisierung"
        ],

        facts: [
            "Ein Tokenizer zerlegt Text in einzelne Bestandteile.",
            "Diese Bestandteile können Wörter, Zahlen und Satzzeichen sein.",
            "Lumora verwendet dafür einen eigenen JavaScript-Tokenizer."
        ]
    },

    render: {
        keywords: [
            "render",
            "hosting",
            "server"
        ],

        facts: [
            "Render kann Node.js-Webserver hosten.",
            "Lumora kann als Node.js-Webservice auf Render laufen."
        ]
    }
};


/* =========================================
   FRAGE ERKENNEN
========================================= */

function detectQuestion(text) {

    const t = normalize(text);

    if (
        t.includes("was ist") ||
        t.includes("was bedeutet") ||
        t.includes("erklär") ||
        t.includes("erklaer")
    ) {
        return "explain";
    }

    if (
        t.includes("wer bist") ||
        t.includes("was bist")
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


/* =========================================
   THEMA ERKENNEN
========================================= */

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


/* =========================================
   ANTWORT SELBST ZUSAMMENSETZEN
========================================= */

function generateOwnAnswer(message) {

    const questionType = detectQuestion(message);
    const topic = detectTopic(message);

    /*
       Keine vorbereitete Standardantwort:
       Lumora baut die Antwort aus mehreren
       Bestandteilen zusammen.
    */

    if (questionType === "identity") {

        const parts = [
            "Ich bin Lumora.",
            "Ich bin eine selbst entwickelte JavaScript-KI.",
            "Mein Text wird zuerst mit tokenizer.js zerlegt.",
            "Danach analysiere ich die Frage und suche nach passenden Informationen."
        ];

        return parts.join(" ");
    }


    if (topic) {

        const data = knowledge[topic];

        if (questionType === "explain") {

            const intro = {
                javascript:
                    "JavaScript ist eine Programmiersprache.",
                tokenizer:
                    "Ein Tokenizer ist ein Werkzeug zur Textanalyse.",
                render:
                    "Render ist eine Plattform, auf der Webservices betrieben werden können."
            };

            const selectedFacts = [...data.facts];

            /*
               Zufällige Reihenfolge = nicht immer exakt
               dieselbe Antwort.
            */
            selectedFacts.sort(() => Math.random() - 0.5);

            return (
                intro[topic] +
                " " +
                selectedFacts.slice(
                    0,
                    Math.min(2, selectedFacts.length)
                ).join(" ")
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
            "Ich habe dazu Informationen über " +
            topic +
            ". " +
            data.facts[
                Math.floor(
                    Math.random() * data.facts.length
                )
            ]
        );
    }


    /* =====================================
       EIGENE ANTWORT AUS TEXT ERSTELLEN
    ===================================== */

    const tokenList = tokenize(message);
    const wordList = words(message);

    if (wordList.length === 0) {
        return "Schreib mir etwas, damit ich es analysieren kann.";
    }

    if (
        containsWord(message, "hallo") ||
        containsWord(message, "hi") ||
        containsWord(message, "hey")
    ) {

        const greetings = [
            "Hallo! Ich habe deine Nachricht analysiert.",
            "Hey! Lumora ist bereit.",
            "Hallo! Mein Tokenizer hat deinen Text bereits zerlegt."
        ];

        return greetings[
            Math.floor(
                Math.random() * greetings.length
            )
        ];
    }


    if (
        containsWord(message, "danke")
    ) {
        return "Gerne! Deine Nachricht wurde erfolgreich analysiert.";
    }


    /*
       Lumora kann auch auf unbekannte Texte
       reagieren, indem sie die erkannte Struktur
       beschreibt.
    */

    if (questionType === "question") {

        return (
            "Ich habe deine Frage erkannt. " +
            "Ich sehe " +
            wordList.length +
            " Wörter und " +
            tokenList.length +
            " Tokens. " +
            "Das passende Wissen habe ich momentan noch nicht gefunden."
        );
    }


    return (
        "Ich habe deinen Text analysiert. " +
        "Er enthält " +
        wordList.length +
        " Wörter und " +
        tokenList.length +
        " Tokens. " +
        "Ich kann mit jedem weiteren Wissenseintrag genauer darauf reagieren."
    );
}


/* =========================================
   CHAT API
========================================= */

app.post("/api/chat", (req, res) => {

    try {

        const message =
            String(req.body?.message || "").trim();

        if (!message) {
            return res.status(400).json({
                error: "Keine Nachricht."
            });
        }

        const tokens = tokenize(message);
        const questionType = detectQuestion(message);
        const topic = detectTopic(message);

        const answer =
            generateOwnAnswer(message);

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


/* =========================================
   TOKENIZER API
========================================= */

app.post("/api/tokenize", (req, res) => {

    const text =
        String(req.body?.text || "");

    res.json({
        tokens: tokenize(text)
    });
});


/* =========================================
   HEALTH
========================================= */

app.get("/api/health", (req, res) => {

    res.json({
        online: true,
        ai: "Lumora",
        tokenizer: true,
        ownAnswerGenerator: true
    });
});


app.listen(
    PORT,
    "0.0.0.0",
    () => {
        console.log("================================");
        console.log("        LUMORA ONLINE");
        console.log("================================");
        console.log("Tokenizer: AKTIV");
        console.log("Antwortgenerator: AKTIV");
        console.log("Port:", PORT);
    }
);
```
